import sys

sys.path.insert(0, "/store/code/ai4science/UIP_EVAL/matsciml")


import argparse
import datetime
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.md import MDLogger
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from experiments.utils.configurator import configurator
from experiments.utils.utils import _get_next_version
from matsciml.interfaces.ase import MatSciMLCalculator
from tqdm import tqdm

from models.matgl_pretrained import load_pretrained_matgl
from models.pretrained_mace import load_pretrained_mace
from utils import (
    get_density,
    minimize_structure,
    replicate_system,
    symmetricize_replicate,
)

time.sleep(random.randint(10, 300))


# Define a function to determine the new interval
def get_new_interval(current_step):
    if current_step < 100:
        return 1
    return 10


def update_completion_file(completions_file):
    def parse_line(line):
        parts = line.split(",")
        return int(parts[0]), datetime.datetime.fromisoformat(parts[1])

    if os.path.isfile(completions_file):
        with open(completions_file, "r") as f:
            lines = f.read().split("\n")
            completed = [parse_line(line) for line in lines if line]
            completed.sort()
            if len(completed) > 0:
                index = completed[-1][0] + 1
            else:
                index = 0
    else:
        completed = []
        index = 0

    current_time = datetime.datetime.now()
    with open(completions_file, "a+") as f:
        f.write(f"{index},{current_time.isoformat()}\n")

    if len(completed) > 1:
        time_diffs = [
            (completed[i][1] - completed[i - 1][1]).total_seconds()
            for i in range(1, len(completed))
        ]
        average_time_diff = sum(time_diffs) / len(time_diffs)
    else:
        average_time_diff = None

    return index, average_time_diff


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

    # Initialize the logger with an initial interval
    initial_interval = get_new_interval(0)
    md_logger = MDLogger(
        dyn,
        init_conf,
        os.path.join(SimDir, "Simulation_thermo.log"),
        header=True,
        stress=True,
        peratom=False,
        mode="w",
    )

    # Attach the logger with the initial interval
    dyn.attach(md_logger, interval=initial_interval)

    # Function to update the logger interval dynamically
    def update_logger_interval():
        current_step = dyn.get_number_of_steps()
        new_interval = get_new_interval(current_step)
        md_logger.interval = new_interval

    update_interval = 10  # Adjust this value as needed
    dyn.attach(update_logger_interval, interval=update_interval)

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
    len_time_list = 0
    time_list = []
    for k in tqdm(range(steps), desc="Running dynamics integration.", total=steps):
        dyn_time_start = time.time()
        dyn.run(1)
        dyn_step_time = time.time() - dyn_time_start

        if len_time_list > 9:
            time_list.pop(0)
            time_list.append(dyn_step_time)
        else:
            time_list.append(dyn_step_time)
            len_time_list = len(time_list)

        counter += 1
        if counter % 100 == 0:
            total_energy = atoms.get_total_energy()
            max_force = np.max(np.abs(atoms.get_forces()))
            wandb.log(
                {
                    "step": k,
                    "density": density[-1],
                    "rolling_avg_step_time": sum(time_list) / 10,
                    "total_energy": total_energy,
                    "max_force": max_force,
                }
            )
        if k < 100:
            write_frame()

    density = np.array(density)
    angles = np.array(angles)
    lattice_parameters = np.array(lattice_parameters)

    # Calculate average values
    avg_density = np.mean(density)
    avg_angles = np.mean(angles, axis=0)
    avg_lattice_parameters = np.mean(lattice_parameters, axis=0)
    return avg_density, avg_angles, avg_lattice_parameters


def get_calculator():
    return MatSciMLCalculator


def get_model(model_name):
    if model_name in ["chgnet_dgl", "m3gnet_dgl"]:
        return load_pretrained_matgl(model_name)
    if model_name in ["mace_pyg"]:
        return load_pretrained_mace(model_name)


def calculator_from_model(args):
    calc = get_calculator()
    model = get_model(args.model_name)
    calc = calc(model, matsciml_model=False)
    return calc


def main(args):
    # wandb.init(
    #     project="md_simulation_chgnet_full",
    #     entity="m3rg",
    #     config=args,
    # )
    wandb.init(
        project="md_simulation_m3gnet_full",
        entity="melo-gonzo",
        config=args,
    )

    sys_info = (subprocess.check_output("lscpu", shell=True).strip()).decode()
    sys_info = sys_info.split("\n")
    try:
        model = [_ for _ in sys_info if "Model name:" in _]
        cpu_type = model[0].split("  ")[-1]
        wandb.log({"cpu_type": cpu_type})
    except Exception:
        pass

    calculator = calculator_from_model(args)
    cif_files_dir = args.input_dir

    dirs = os.listdir(cif_files_dir)
    dirs.sort()

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

            wandb.log({"num_atoms": atoms.positions.shape[0]})

            # Minimize the structure
            atoms.calc = calculator
            minimize_time_start = time.time()
            atoms = minimize_structure(atoms, steps=args.minimize_steps)
            relaxation_time = time.time() - minimize_time_start
            wandb.log({"relaxation_time": relaxation_time})
            # Calculate density and cell lengths and angles
            density = get_density(atoms)
            cell_lengths_and_angles = atoms.get_cell_lengths_and_angles().tolist()
            sim_dir = os.path.join(args.results_dir, f"{args.index}_Simulation_{file}")
            print("SIMDIR:", sim_dir)
            os.makedirs(sim_dir, exist_ok=True)
            simulation_time_start = time.time()
            avg_density, avg_angles, avg_lattice_parameters = run_simulation(
                calculator,
                atoms,
                pressure=Press,
                temperature=Temp,
                timestep=args.timestep,
                steps=args.runsteps,
                SimDir=sim_dir,
            )
            simulation_time = time.time() - simulation_time_start
            wandb.log({"simulation_time": simulation_time})
            print(avg_density)
            # Append the results to the data list
            data.append(
                [file[:-4], density]
                + cell_lengths_and_angles
                + [avg_density]
                + avg_lattice_parameters.tolist()
                + avg_angles.tolist()
            )
            # Log final results to wandb
            total_energy = atoms.get_total_energy()
            max_force = np.max(np.abs(atoms.get_forces()))
            wandb.log(
                {
                    "exp_density": density,
                    "avg_density": avg_density,
                    "final_total_energy": total_energy,
                    "final_max_force": max_force,
                }
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

    # Finish the wandb run
    wandb.finish()


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
    parser.add_argument("--minimize_steps", type=int, default=1000)
    parser.add_argument("--thermo_interval", type=int, default=10)
    parser.add_argument("--log_dir_base", type=Path, default="./simulation_results")
    parser.add_argument("--replica", action="store_true")

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
    args.experiment_times_file = "./k8/experiment_times.txt"

    if args.replica:
        # time.sleep(random.randint(10, 300))
        completions_file = "./k8/completed.txt"
        args.index, args.avg_completion_time = update_completion_file(completions_file)

    if args.index > 2684:
        time.sleep(100000)
        os._exit(0)

    configurator.configure_models(args.model_config)
    configurator.configure_datasets(args.dataset_config)
    configurator.configure_trainer(args.trainer_config)
    log_dir_base = args.log_dir_base.joinpath(args.model_name, str(args.index))
    results_dir = log_dir_base.joinpath(_get_next_version(log_dir_base))
    results_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir = results_dir

    with open(results_dir.joinpath("cli_args.yaml"), "w") as f:
        command = "python experimental.py " + " ".join(
            f"--{k} {v}" for k, v in vars(args).items()
        )
        args.command = command
        yaml.safe_dump({k: str(v) for k, v in args.__dict__.items()}, f, indent=2)

    with open(results_dir.joinpath("cpu_spec.txt"), "w") as f:
        result = subprocess.run(
            "lscpu", shell=True, stdout=f, stderr=subprocess.PIPE, text=True
        )

    try:
        total_time_start = time.time()
        main(args)
        total_time_end = time.time() - total_time_start
        with open(args.experiment_times_file, "a+") as f:
            f.write(str(total_time_end) + "\n")
    except Exception:
        import traceback

        traceback.format_exc()
        with open(results_dir.joinpath("error.txt"), "w") as f:
            f.write("\n" + str(traceback.format_exc()))
            print(traceback.format_exc())
