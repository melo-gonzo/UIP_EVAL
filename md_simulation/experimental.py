# import sys

# sys.path.insert(0, "/store/code/ai4science/matsciml")

import argparse
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Callable

import dgl
import numpy as np
import pandas as pd
import torch
import yaml
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.md import MDLogger
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from einops import reduce
from experiments.utils.configurator import configurator
from experiments.utils.utils import _get_next_version, instantiate_arg_dict
from models.matgl_pretrained import load_matgl
from models.pretrained_mace import load_mace
from tqdm import tqdm
from utils import (
    get_density,
    minimize_structure,
    replicate_system,
    symmetricize_replicate,
)

from matsciml.common.types import AbstractGraph, Embeddings
from matsciml.interfaces.ase import MatSciMLCalculator
from matsciml.models.base import ForceRegressionTask, dynamic_gradients_context
from matsciml.models.utils.io import multitask_from_checkpoint


class ForceRegressionTask(ForceRegressionTask):
    def forward(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        # for ease of use, this task will always compute forces
        # del batch["embeddings"]
        with dynamic_gradients_context(True, self.has_rnn):
            # first ensure that positions tensor is backprop ready
            if "graph" in batch:
                graph = batch["graph"]
                cell = batch["cell"]
                # the DGL case
                if hasattr(graph, "ndata"):
                    pos: torch.Tensor = graph.ndata.get("pos")

                    # for frame averaging
                    fa_rot = graph.ndata.get("fa_rot", None)
                    fa_pos = graph.ndata.get("fa_pos", None)
                    graph.ndata["pos"] = pos
                else:
                    # otherwise assume it's PyG
                    pos: torch.Tensor = graph.pos

                    # for frame averaging
                    fa_rot = getattr(graph, "fa_rot", None)
                    fa_pos = getattr(graph, "fa_pos", None)
                    cell = getattr(graph, "cell", None)
            else:
                graph = None
                # assume point cloud otherwise
                pos: torch.Tensor = batch.get("pos")
                # no frame averaging architecture yet for point clouds
                fa_rot = None
                fa_pos = None
            if pos is None:
                raise ValueError(
                    "No atomic positions were found in batch - neither as standalone tensor nor graph.",
                )
            if isinstance(pos, torch.Tensor):
                pos.requires_grad_(True)
                displacement = torch.zeros(
                    (1, 3, 3),
                    dtype=pos.dtype,
                    device=pos.device,
                )

                displacement.requires_grad_(True)
                symmetric_displacement = 0.5 * (
                    displacement + displacement.transpose(-1, -2)
                )  # From https://github.com/mir-group/nequip
                pos = pos + torch.einsum(
                    "be,bec->bc",
                    pos,
                    symmetric_displacement,
                )
                if "graph" in batch:
                    graph.pos = pos
                if hasattr(graph, "ndata"):
                    graph.ndata["pos"] = pos

                if fa_pos is not None:
                    for k in range(len(fa_pos)):
                        fa_pos[0].requires_grad_(True)
                        fa_pos[0] = fa_pos[0] + torch.einsum(
                            "be,bec->bc",
                            pos,
                            symmetric_displacement,
                        )

            elif isinstance(pos, list):
                [p.requires_grad_(True) for p in pos]
            else:
                raise ValueError(
                    f"'pos' data is required for force calculation, but isn't a tensor or a list of tensors: {type(pos)}.",
                )
            if isinstance(fa_pos, torch.Tensor):
                fa_pos.requires_grad_(True)
            elif isinstance(fa_pos, list):
                [f_p.requires_grad_(True) for f_p in fa_pos]
            if "embeddings" in batch:
                embeddings = batch.get("embeddings")
            else:
                embeddings = self.encoder(batch)

            natoms = batch.get("natoms", None)
            outputs = self.process_embedding(
                embeddings, pos, displacement, cell, fa_rot, fa_pos, natoms, graph
            )
        return outputs

    def process_embedding(
        self,
        embeddings: Embeddings,
        pos: torch.Tensor,
        displacement: torch.Tensor,
        cell: torch.Tensor,
        fa_rot: None | torch.Tensor = None,
        fa_pos: None | torch.Tensor = None,
        natoms: None | torch.Tensor = None,
        graph: None | AbstractGraph = None,
    ) -> dict[str, torch.Tensor]:
        outputs = {}

        # compute node-level contributions to the energy
        node_energies = self.output_heads["energy"](embeddings.point_embedding)
        # figure out how we're going to reduce node level energies
        # depending on the representation and/or the graph framework
        if graph is not None:
            if isinstance(graph, dgl.DGLGraph):
                graph.ndata["node_energies"] = node_energies

                def readout(node_energies: torch.Tensor):
                    return dgl.readout_nodes(
                        graph, "node_energies", op=self.embedding_reduction_type
                    )

            else:
                # assumes a batched pyg graph
                batch = graph.batch
                from torch_geometric.utils import scatter

                def readout(node_energies: torch.Tensor):
                    return scatter(
                        node_energies,
                        batch,
                        dim=-2,
                        reduce=self.embedding_reduction_type,
                    )

        else:

            def readout(node_energies: torch.Tensor):
                return reduce(
                    node_energies, "b ... d -> b ()", self.embedding_reduction_type
                )

        def energy_and_force(
            pos: torch.Tensor,
            displacement: torch.Tensor,
            cell: torch.Tensor,
            node_energies: torch.Tensor,
            readout: Callable,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # we sum over points and keep dimension as 1
            energy = readout(node_energies)
            if energy.ndim == 1:
                energy.unsqueeze(-1)
            # now use autograd for force calculation

            # force = (
            #     -1
            #     * torch.autograd.grad(
            #         energy,
            #         pos,
            #         grad_outputs=torch.ones_like(energy),
            #         create_graph=True,
            #     )[0]
            # )

            forces, virials = torch.autograd.grad(
                outputs=[energy],  # [n_graphs, ]
                inputs=[pos, displacement],  # [n_nodes, 3]
                retain_graph=True,  # Make sure the graph is not destroyed during training
                create_graph=True,  # Create graph for second derivative
                allow_unused=True,
            )

            cell = cell.view(-1, 3, 3)
            volume = torch.einsum(
                "zi,zi->z",
                cell[:, 0, :],
                torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
            ).unsqueeze(-1)
            stress = virials / volume.view(-1, 1, 1)

            return energy, -1 * forces, stress

        # not using frame averaging
        if fa_pos is None:
            energy, force, stress = energy_and_force(
                pos, displacement, cell, node_energies, readout
            )
        else:
            energy = []
            force = []
            stress = []
            for idx, pos in enumerate(fa_pos):
                frame_embedding = node_energies[:, idx, :]
                frame_energy, frame_force, frame_stress = energy_and_force(
                    pos, displacement, cell, frame_embedding, readout
                )
                force.append(frame_force)
                energy.append(frame_energy.unsqueeze(-1))
                stress.append(frame_stress)

        # check to see if we are frame averaging
        if fa_rot is not None:
            all_forces = []
            # loop over each frame prediction, and transform to guarantee
            # equivariance of frame averaging method
            natoms = natoms.squeeze(-1).to(int)
            for frame_idx, frame_rot in enumerate(fa_rot):
                repeat_rot = torch.repeat_interleave(
                    frame_rot,
                    natoms,
                    dim=0,
                ).to(self.device)
                rotated_forces = (
                    force[frame_idx].view(-1, 1, 3).bmm(repeat_rot.transpose(1, 2))
                )
                all_forces.append(rotated_forces)
            # combine all the force and energy data into a single tensor
            # using frame averaging, the expected shapes after concatenation are:
            # force - [num positions, num frames, 3]
            # energy - [batch size, num frames, 1]
            force = torch.cat(all_forces, dim=1)
            energy = torch.cat(energy, dim=1)
            stress = torch.cat(stress, dim=1)
        # reduce outputs to what are expected shapes
        outputs["force"] = reduce(
            force,
            "n ... d -> n d",
            self.embedding_reduction_type,
            d=3,
        )
        # this may not do anything if we aren't frame averaging
        # since the reduction is also done in the energy_and_force call
        outputs["energy"] = reduce(
            energy,
            "b ... d -> b d",
            self.embedding_reduction_type,
            d=1,
        )

        # this ensures that we get a scalar value for every node
        # representing the energy contribution
        outputs["node_energies"] = node_energies
        outputs["stress"] = stress
        return outputs


def run_simulation(
    calculator: Calculator,
    atoms: Atoms,
    pressure: float = 0.000101325,  # GPa
    temperature: float = 298,
    timestep: float = 0.1,
    steps: int = 10,
    SimDir: str | Path = Path.cwd(),
):
    # Define the temperature and pressure
    init_conf = atoms
    init_conf.set_calculator(calculator)
    # Initialize the NPT dynamics
    MaxwellBoltzmannDistribution(init_conf, temperature_K=temperature)

    dyn = NPTBerendsen(
        init_conf,
        timestep=timestep * units.fs,
        temperature_K=temperature,
        pressure_au=pressure * units.bar,
        compressibility_au=4.57e-5 / units.bar,
    )

    dyn.attach(
        MDLogger(
            dyn,
            init_conf,
            os.path.join(SimDir, "Simulation_thermo.log"),
            header=True,
            stress=True,
            peratom=False,
            mode="w",
        ),
        interval=args.thermo_interval,
    )

    density = []
    angles = []
    lattice_parameters = []

    def write_frame():
        dyn.atoms.write(
            os.path.join(SimDir, f"MD_{atoms.get_chemical_formula()}_NPT.xyz"),
            append=True,
        )

        cell = dyn.atoms.get_cell()

        lattice_parameters.append(cell.lengths())  # Get the lattice parameters
        angles.append(cell.angles())  # Get the angles
        density.append(get_density(atoms))

    dyn.attach(write_frame, interval=args.trajdump_interval)

    counter = 0
    for k in tqdm(range(steps), desc="Running dynamics integration.", total=steps):
        dyn.run(1)
        counter += 1

    density = np.array(density)
    angles = np.array(angles)
    lattice_parameters = np.array(lattice_parameters)

    # Calculate average values
    avg_density = np.mean(density)
    avg_angles = np.mean(angles, axis=0)
    avg_lattice_parameters = np.mean(lattice_parameters, axis=0)
    return avg_density, avg_angles, avg_lattice_parameters


def calculator_from_model(args):
    checkpoint = args.model_path
    model_args = instantiate_arg_dict(deepcopy(configurator.models[args.model_name]))
    if args.task == "ForceRegressionTask":
        if args.model_name in ["chgnet_dgl", "m3gnet_dgl"]:
            model = load_matgl(checkpoint)
            model = model.to(torch.double)
            calc = MatSciMLCalculator(
                model, transforms=model_args["transforms"], from_matsciml=False
            )
        elif "mace" not in args.model_name:
            calc = MatSciMLCalculator.from_pretrained_force_regression(
                args.model_path, transforms=model_args["transforms"]
            )
        else:
            model = load_mace(checkpoint)
            calc = MatSciMLCalculator(model, transforms=model_args["transforms"])

    if args.task == "MultiTaskLitModule":
        model = multitask_from_checkpoint(checkpoint)

    return calc


def main(args):
    calculator = calculator_from_model(args)
    cif_files_dir = args.input_dir

    dirs = os.listdir(cif_files_dir)

    folder = dirs[args.index]
    with open(results_dir.joinpath("cli_args.yaml"), "a") as f:
        yaml.safe_dump({"file_name": folder}, f, indent=2)
    print("reading_folder number:", folder)

    # List to hold the data
    data = []
    folder_path = os.path.join(cif_files_dir, folder)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            Temp, Press = file.split("_")[2:4]
            Temp, Press = float(Temp), float(Press)

            atoms = read(file_path)

            # Replicate_system
            replication_factors, size = symmetricize_replicate(
                len(atoms),
                max_atoms=args.max_atoms,
                box_lengths=atoms.get_cell_lengths_and_angles()[:3],
            )
            atoms = replicate_system(atoms, replication_factors)

            # Minimize the structure
            atoms.set_calculator(calculator)
            atoms = minimize_structure(atoms)

            # Calculate density and cell lengths and angles
            density = get_density(atoms)
            cell_lengths_and_angles = atoms.get_cell_lengths_and_angles().tolist()
            sim_dir = os.path.join(args.results_dir, f"{args.index}_Simulation_{file}")
            print("SIMDIR:", sim_dir)
            os.makedirs(sim_dir, exist_ok=True)
            avg_density, avg_angles, avg_lattice_parameters = run_simulation(
                calculator,
                atoms,
                pressure=Press,
                temperature=Temp,
                timestep=args.timestep,
                steps=args.runsteps,
                SimDir=sim_dir,
            )
            print(avg_density)
            # Append the results to the data list
            data.append(
                [file[:-4], density]
                + cell_lengths_and_angles
                + [avg_density]
                + avg_lattice_parameters.tolist()
                + avg_angles.tolist()
            )
            # Create a DataFrame
            columns = [
                "Filename",
                "Exp_Density (g/cm³)",
                "Exp_a (Å)",
                "Exp_b (Å)",
                "Exp_c (Å)",
                "Exp_alpha (°)",
                "Exp_beta (°)",
                "Exp_gamma (°)",
                "Sim_Density (g/cm³)",
                "Sim_a (Å)",
                "Sim_b (Å)",
                "Sim_c (Å)",
                "Sim_alpha (°)",
                "Sim_beta (°)",
                "Sim_gamma (°)",
            ]
            df = pd.DataFrame(data, columns=columns)

            # Save the DataFrame to a CSV file
            df.to_csv(os.path.join(sim_dir, "Data.csv"), index=False)


if __name__ == "__main__":
    # Seed for the Python random module
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        torch.cuda.manual_seed_all(123)  # if you are using multi-GPU.
    parser = argparse.ArgumentParser(description="Run MD simulation")
    parser.add_argument("--index", type=int, default=0, help="index of folder")
    parser.add_argument("--runsteps", type=int, default=50_000)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_atoms", type=int, default=100)
    parser.add_argument("--trajdump_interval", type=int, default=10)
    parser.add_argument("--minimize_steps", type=int, default=200)
    parser.add_argument("--thermo_interval", type=int, default=10)
    parser.add_argument("--log_dir_base", type=Path, default="./simulation_results")
    parser.add_argument("--task", default="ForceRegressionTask")

    parser.add_argument(
        "--dataset_config",
        type=Path,
        default=Path("/store/code/ai4science/matsciml/experiments").joinpath(
            "configs", "datasets"
        ),
        help="Dataset config folder or yaml file to use.",
    )
    parser.add_argument(
        "--trainer_config",
        type=Path,
        default=Path("/store/code/ai4science/matsciml/experiments").joinpath(
            "configs", "trainer"
        ),
        help="Trainer config folder or yaml file to use.",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        default=Path("/store/code/ai4science/matsciml/experiments").joinpath(
            "configs", "models"
        ),
        help="Model config folder or yaml file to use.",
    )
    args = parser.parse_args()
    configurator.configure_models(args.model_config)
    configurator.configure_datasets(args.dataset_config)
    configurator.configure_trainer(args.trainer_config)
    log_dir_base = args.log_dir_base.joinpath(
        args.model_name, args.task, str(args.index)
    )
    results_dir = log_dir_base.joinpath(_get_next_version(args.log_dir_base))
    results_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir = results_dir
    with open(results_dir.joinpath("cli_args.yaml"), "w") as f:
        command = "python mp_test_runner.py " + " ".join(
            f"--{k} {v}" for k, v in vars(args).items()
        )
        args.command = command
        yaml.safe_dump({k: str(v) for k, v in args.__dict__.items()}, f, indent=2)
    try:
        main(args)
    except Exception:
        import traceback

        with open(results_dir.joinpath("error.txt"), "w") as f:
            f.write("\n" + str(traceback.format_exc()))
            print(traceback.format_exc())
