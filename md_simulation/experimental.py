import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from ase import Atoms, units
from ase.io import read, write
from ase.md import MDLogger
from ase.md.langevin import Langevin

# torch.set_default_dtype(torch.float64)
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import (
    BFGS,  # Import BFGS optimizer
    FIRE,
)
from loguru import logger
from mace.calculators import MACECalculator
from matsciml.preprocessing.atoms_to_graphs import *
from tqdm import tqdm

from checkpoint import *
from checkpoint import multitask_from_checkpoint

# sys.path.append("/home/civil/phd/cez218288/scratch/mace_v_0.3.5/mace")  # Path to mace directory relative to MDBENCHGNN Folder
from Utils import *


def get_density(atoms):
    amu_to_grams = 1.66053906660e-24  # 1 amu = 1.66053906660e-24 grams
    angstrom_to_cm = 1e-8  # 1 Å = 1e-8 cm
    mass_amu = atoms.get_masses().sum()
    mass_g = (
        mass_amu * amu_to_grams
    )  # Get the volume of the atoms object in cubic angstroms (Å³)
    volume_A3 = atoms.get_volume()
    volume_cm3 = volume_A3 * (angstrom_to_cm**3)  # 1 Å³ = 1e-24 cm³
    density = mass_g / volume_cm3

    return density


def write_xyz(Filepath, atoms):
    """Writes ovito xyz file"""
    R = atoms.get_positions()
    species = atoms.get_atomic_numbers()
    cell = atoms.get_cell()

    with open(Filepath, "w") as f:
        f.write(str(R.shape[0]) + "\n")
        flat_cell = cell.flatten()
        f.write(
            f'Lattice="{flat_cell[0]} {flat_cell[1]} {flat_cell[2]} {flat_cell[3]} {flat_cell[4]} {flat_cell[5]} {flat_cell[6]} {flat_cell[7]} {flat_cell[8]}" Properties=species:S:1:pos:R:3 Time=0.0'
        )
        for i in range(R.shape[0]):
            f.write(
                "\n"
                + str(species[i])
                + "\t"
                + str(R[i, 0])
                + "\t"
                + str(R[i, 1])
                + "\t"
                + str(R[i, 2])
            )


def replicate_system(atoms, replicate_factors):
    """
    Replicates the given ASE Atoms object according to the specified replication factors.
    """
    nx, ny, nz = replicate_factors
    original_cell = atoms.get_cell()
    original_positions = (
        atoms.get_scaled_positions() @ original_cell
    )  # Scaled or Unscaled ?
    original_numbers = atoms.get_atomic_numbers()
    x_cell, y_cell, z_cell = original_cell[0], original_cell[1], original_cell[2]
    new_numbers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                new_numbers += [original_numbers]
    pos_after_x = np.concatenate([original_positions + i * x_cell for i in range(nx)])
    pos_after_y = np.concatenate([pos_after_x + i * y_cell for i in range(ny)])
    pos_after_z = np.concatenate([pos_after_y + i * z_cell for i in range(nz)])
    new_cell = [nx * original_cell[0], ny * original_cell[1], nz * original_cell[2]]
    new_atoms = Atoms(
        numbers=np.concatenate(new_numbers),
        positions=pos_after_z,
        cell=new_cell,
        pbc=atoms.get_pbc(),
    )
    return new_atoms


def Symmetricize_replicate(curr_atoms, max_atoms, box_lengths):
    replication = [1, 1, 1]
    atom_count = curr_atoms
    lengths = box_lengths
    while atom_count < (max_atoms // 2):
        direction = np.argmin(box_lengths)
        replication[direction] += 1
        lengths[direction] = box_lengths[direction] * replication[direction]
        atom_count = curr_atoms * replication[0] * replication[1] * replication[2]
    return replication, atom_count
    # Create new Atoms object


def minimize_structure(atoms, fmax=0.05, steps=10):
    """
    Perform energy minimization on the given ASE Atoms object using the FIRE optimizer.

    Parameters:
    atoms (ase.Atoms): The Atoms object to be minimized.
    fmax (float): The maximum force tolerance for the optimization (default: 0.01 eV/Å).
    steps (int): The maximum number of optimization steps (default: 1000).

    Returns:
    ase.Atoms: The minimized Atoms object.
    """
    dyn = FIRE(atoms, trajectory=None)
    dyn.run(fmax=fmax, steps=steps)
    return atoms


class TestArgs:
    runsteps = 50000
    model_name = "tensornet"  ##[mace, faenet, tensornet]
    model_path = "/home/m3rg2000/Simulation/checkpoints-2024/epoch=4-step=4695_tensornet_force_r.ckpt"
    timestep = 1.0
    results_dir = "/home/m3rg2000/Universal_matscimal/Sim_output"
    input_dir = "/home/m3rg2000/Universal_matscimal/Data/exp_1"
    device = "cuda"
    max_atoms = 100  # Replicate upto max_atoms (Min. will be max_atoms/2) (#Won't reduce if more than max_atoms)
    trajdump_interval = 10
    minimize_steps = 200
    thermo_interval = 10


config = TestArgs()


def run_simulation(
    calculator,
    atoms,
    pressure=0.000101325,
    temperature=298,
    timestep=0.1,
    steps=10,
    SimDir="./",
):  ##pressure=0.000101325 GPa
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
        interval=config.thermo_interval,
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

    dyn.attach(write_frame, interval=config.trajdump_interval)

    counter = 0
    for k in tqdm(range(steps)):
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


def main(args, config):
    start_time = time.time()
    Loaded_model = multitask_from_checkpoint(config.model_path)
    calculator = ASEcalculator(Loaded_model, config.model_name)
    # calculator = MACECalculator(model_paths=config.model_path, device=config.device, default_dtype='float64')
    # calculator.model.double()  # Change model weights type to double precision (hack to avoid error)
    model_name = "mace"
    cif_files_dir = config.input_dir
    # output_file = config.out_dir
    import pandas as pd
    from tqdm import tqdm

    Dirs = os.listdir(cif_files_dir)
    # for k in range(len(Dirs)):
    #     print(k,Dirs[k])
    folder = Dirs[args.index]
    print("readong_folder number:", folder)

    # List to hold the data
    data = []
    folder_path = os.path.join(cif_files_dir, folder)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            Temp, Press = file.split("_")[2:4]
            Temp, Press = float(Temp), float(Press)
            # TrajPath=os.path.join(config.traj_folder,"_".join(file.split("_")[:2])+'_Trajectory.xyz')

            # try:
            atoms = read(file_path)

            # Replicate_system
            replication_factors, size = Symmetricize_replicate(
                len(atoms),
                max_atoms=config.max_atoms,
                box_lengths=atoms.get_cell_lengths_and_angles()[:3],
            )
            atoms = replicate_system(atoms, replication_factors)

            # Minimize the structure
            atoms.set_calculator(calculator)
            atoms = minimize_structure(atoms)

            # Calculate density and cell lengths and angles
            density = get_density(atoms)
            cell_lengths_and_angles = atoms.get_cell_lengths_and_angles().tolist()
            SimDir = os.path.join(config.results_dir, f"{args.index}_Simulation_{file}")
            print("SIMDIR:", SimDir)
            os.makedirs(SimDir, exist_ok=True)
            avg_density, avg_angles, avg_lattice_parameters = run_simulation(
                calculator,
                atoms,
                pressure=Press,
                temperature=Temp,
                timestep=config.timestep,
                steps=config.runsteps,
                SimDir=SimDir,
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
            df.to_csv(os.path.join(SimDir, f"Data.csv"), index=False)
            # print(f"Data saved to {output_file}")
            # except:
            #     print("filename", file_path)

    # Create a DataFrame
    # columns = ["Filename", "Exp_Density (g/cm³)", "Exp_a (Å)", "Exp_b (Å)", "Exp_c (Å)", "Exp_alpha (°)", "Exp_beta (°)", "Exp_gamma (°)"
    #            ,"Sim_Density (g/cm³)", "Sim_a (Å)", "Sim_b (Å)", "Sim_c (Å)", "Sim_alpha (°)", "Sim_beta (°)", "Sim_gamma (°)"]
    # df = pd.DataFrame(data, columns=columns)

    # # Save the DataFrame to a CSV file
    # df.to_csv(output_file, index=False)
    # print(f"Data saved to {output_file}")


if __name__ == "__main__":
    config = TestArgs()
    # Seed for the Python random module
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        torch.cuda.manual_seed_all(123)  # if you are using multi-GPU.
    parser = argparse.ArgumentParser(description="Run MD simulation with MACE model")
    parser.add_argument("--index", type=int, default=0, help="index of folder")
    # parser.add_argument("--init_conf_path", type=str, default="example/lips20/data/test/botnet.xyz", help="Path to the initial configuration")
    # parser.add_argument("--device", type=str, default="cuda", help="Device: ['cpu', 'cuda']")
    # parser.add_argument("--input_dir", type=str, default="./", help="folder path")
    # parser.add_argument("--out_dir", type=str, default="out_dir_sl/neqip/lips20/exp.csv", help="Output path")
    # parser.add_argument("--results_dir", type=str, default="out_dir_sl/neqip/lips20/", help="Output  directory path")

    # parser.add_argument("--temp", type=float, default=300, help="Temperature in Kelvin")
    # parser.add_argument("--pressure", type=float, default=1, help="pressure in atm")
    # parser.add_argument("--timestep", type=float, default=1.0, help="Timestep in fs units")
    # parser.add_argument("--runsteps", type=int, default=1000, help="No. of steps to run")
    # parser.add_argument("--sys_name", type=str, default='System', help="System name")
    # parser.add_argument("--traj_folder", type=str, default="/home/civil/phd/cez218288/Benchmarking/MDBENCHGNN/mace_universal_2.0/EXP/Quartz/a.xyz")

    args = parser.parse_args()
    main(args, config)
