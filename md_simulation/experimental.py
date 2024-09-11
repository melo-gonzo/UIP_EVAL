from copy import deepcopy
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.md import MDLogger
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from experiments.utils.utils import _get_next_version
from matsciml.models.utils.io import multitask_from_checkpoint
from tqdm import tqdm
from utils import (
    get_density,
    minimize_structure,
    replicate_system,
    symmetricize_replicate,
)

from matsciml.interfaces.ase import MatSciMLCalculator
from experiments.utils.configurator import configurator
from experiments.utils.utils import instantiate_arg_dict

from models.matgl_pretrained import load_matgl
from models.pretrained_mace import load_mace


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
    log_dir_base = args.log_dir_base.joinpath(args.model_name, args.task)
    results_dir = log_dir_base.joinpath(_get_next_version(args.log_dir_base))
    results_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir = results_dir

    main(args)
