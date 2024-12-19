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
import torch
import wandb
import yaml
from ase.io import read
from experiments.utils.configurator import configurator
from experiments.utils.utils import _get_next_version
from loguru import logger
from matsciml.interfaces.ase import MatSciMLCalculator

from models.matgl_pretrained import load_pretrained_matgl
from models.pretrained_mace import load_pretrained_mace
from npt_simulation import run


HACKED_REPLICA = False
if HACKED_REPLICA:
    time.sleep(random.randint(10, 300))


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


def setup_logger(
    project: str,
    entity: str,
    config=None,
) -> None:
    wandb.init(
        project=project,
        entity=entity,
        config=config,
    )


def log_hardware_environment():
    sys_info = (subprocess.check_output("lscpu", shell=True).strip()).decode()
    sys_info = sys_info.split("\n")
    try:
        model = [_ for _ in sys_info if "Model name:" in _]
        cpu_type = model[0].split("  ")[-1]
        if not args.debug:
            wandb.log({"cpu_type": cpu_type})
    except Exception:
        pass


def dump_cli_args(source_folder):
    with open(results_dir.joinpath("cli_args.yaml"), "a") as f:
        yaml.safe_dump({"file_name": source_folder}, f, indent=2)


def get_source_folder(args):
    cif_files_dir = args.input_dir
    dirs = os.listdir(cif_files_dir)
    dirs.sort()
    source_folder = dirs[args.index]
    source_folder_path = os.path.join(cif_files_dir, source_folder)
    logger.info("Reading folder number:", source_folder)
    return source_folder, source_folder_path


def main(args):
    wandb_setup = {"project": args.project, "entity": args.entity, "config": args}
    if not args.debug:
        setup_logger(**wandb_setup)

    log_hardware_environment()

    source_folder, source_folder_path = get_source_folder(args)
    dump_cli_args(source_folder=source_folder)

    calculator = calculator_from_model(args)

    assert os.path.isdir(
        source_folder_path
    ), f"Source folder path is not a directory: {source_folder_path}"

    for file in os.listdir(source_folder_path):
        file_path = os.path.join(source_folder_path, file)
        temperature, pressure = file.split("_")[2:4]
        temperature, pressure = float(temperature), float(pressure)

        atoms = read(file_path)
        atoms.calc = calculator
        run(atoms, args, temperature, pressure, file)

    if not args.debug:
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
    parser.add_argument("--project", type=str, default="debug")
    parser.add_argument("--entity", type=str, default="melo-gonzo")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--dataset_config",
        type=Path,
        default=Path("/store/code/ai4science/UIP_EVAL/matsciml/experiments").joinpath(
            "configs", "datasets"
        ),
        help="Dataset config folder or yaml file to use.",
    )
    parser.add_argument(
        "--trainer_config",
        type=Path,
        default=Path("/store/code/ai4science/UIP_EVAL/matsciml/experiments").joinpath(
            "configs", "trainer"
        ),
        help="Trainer config folder or yaml file to use.",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        default=Path("/store/code/ai4science/UIP_EVAL/matsciml/experiments").joinpath(
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
    if args.debug:
        args.results_dir = "./debug_logs"

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
