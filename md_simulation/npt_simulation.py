import sys

sys.path.insert(0, "/store/code/ai4science/UIP_EVAL/matsciml")


import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md import MDLogger
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tqdm import tqdm

from utils import (
    get_density,
    minimize_structure,
    replicate_system,
    symmetricize_replicate,
)

from loguru import logger


DEBUG = True


# Define a function to determine the new interval
def get_new_interval(current_step):
    if current_step < 100:
        return 1
    return 10


def run_simulation(
    calculator: Calculator,
    atoms: Atoms,
    pressure: float = 0.000101325,  # GPa
    temperature: float = 298,
    timestep: float = 0.1,
    steps: int = 10,
    SimDir: str | Path = Path.cwd(),
    traj_dump_interval: int = 10,
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

    dyn.attach(write_frame, interval=traj_dump_interval)

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
            if not DEBUG:
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


def run_relaxation(atoms, minimize_steps):
    minimize_time_start = time.time()
    atoms = minimize_structure(atoms, steps=minimize_steps)
    relaxation_time = time.time() - minimize_time_start
    if not DEBUG:
        wandb.log({"relaxation_time": relaxation_time})
    # Calculate density and cell lengths and angles
    density = get_density(atoms)
    cell_params = atoms.get_cell_lengths_and_angles().tolist()
    return atoms, density, cell_params


def run(atoms, args, temperature, pressure, file):
    data = []
    # Replicate_system
    replication_factors, size = symmetricize_replicate(
        len(atoms),
        max_atoms=args.max_atoms,
        box_lengths=atoms.get_cell_lengths_and_angles()[:3],
    )
    atoms = replicate_system(atoms, replication_factors)

    if not DEBUG:
        wandb.log({"num_atoms": atoms.positions.shape[0]})

    # Minimize the structure
    atoms, density, cell_params = run_relaxation(atoms, args.minimize_steps)

    sim_dir = os.path.join(args.results_dir, f"{args.index}_Simulation_{file}")
    logger.info(f"Simulation directory: {sim_dir}")

    os.makedirs(sim_dir, exist_ok=True)
    simulation_time_start = time.time()
    avg_density, avg_angles, avg_lattice_parameters = run_simulation(
        atoms.calc,
        atoms,
        pressure=pressure,
        temperature=temperature,
        timestep=args.timestep,
        steps=args.runsteps,
        SimDir=sim_dir,
        traj_dump_interval=args.trajdump_interval,
    )
    simulation_time = time.time() - simulation_time_start
    if not DEBUG:
        wandb.log({"simulation_time": simulation_time})
    # Append the results to the data list
    data.append(
        [file[:-4], density]
        + cell_params
        + [avg_density]
        + avg_lattice_parameters.tolist()
        + avg_angles.tolist()
    )
    # Log final results to wandb
    total_energy = atoms.get_total_energy()
    max_force = np.max(np.abs(atoms.get_forces()))
    if not DEBUG:
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
