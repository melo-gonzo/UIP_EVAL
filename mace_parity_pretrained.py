import random
import traceback
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import os
from mace.calculators import mace_mp
from pymatgen.core.structure import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from matsciml.lightning.data_utils import MatSciMLDataModule


def set_random_seeds(seed_value=123):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


class MaterialPropertyPredictor:
    def __init__(
        self, model_name, dataset_name, test_split_path, batch_size, max_samples=None
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.test_split_path = test_split_path
        self.batch_size = batch_size
        self.calc = mace_mp(model="large", default_dtype="float64", device="cuda")
        self.adaptor = AseAtomsAdaptor()
        self.dm = MatSciMLDataModule(
            "MaterialsProjectDataset",
            train_path=self.test_split_path,
            batch_size=self.batch_size,
        )
        self.dm.setup()
        if max_samples is None:
            max_samples = len(self.dm.train_dataloader())
        self.max_samples = max_samples

    def process_data_loader(self, data_loader):
        results = self.initialize_prediction_lists()
        counter = 0

        for batch in tqdm(data_loader, total=len(data_loader)):
            try:
                atoms = self.atoms_from_batch(batch)
                self.calc.calculate(atoms)

                predicted_energy = self.calc.results["energy"]
                predicted_forces = self.calc.results["forces"]

                target_energy = (
                    batch["targets"]["corrected_total_energy"].cpu().numpy().item()
                )
                target_force = batch["targets"]["force"].cpu().numpy()

                self.update_results(
                    results,
                    predicted_energy,
                    target_energy,
                    predicted_forces,
                    target_force,
                )

                counter += 1
                if counter >= self.max_samples:
                    break
            except Exception:
                traceback.print_exc()

        return results

    @staticmethod
    def initialize_prediction_lists():
        return {
            "predicted_energy": [],
            "target_energy": [],
            "predicted_fx": [],
            "target_fx": [],
            "predicted_fy": [],
            "target_fy": [],
            "predicted_fz": [],
            "target_fz": [],
        }

    def atoms_from_batch(self, batch):
        coords = batch["pos"]
        lattice = batch["lattice_features"]["lattice_params"].squeeze()
        atomic_numbers = batch["atomic_numbers"].squeeze()
        angles = lattice[3:] * (180 / (2 * torch.pi))
        lattice = Lattice.from_parameters(
            lattice[0],
            lattice[1],
            lattice[2],
            angles[0],
            angles[1],
            angles[2],
            pbc=(True, True, True),
        )
        if coords.max() > 1.0 or coords.min() < 0.0:
            is_frac = False
        else:
            is_frac = True

        structure = Structure(
            lattice,
            atomic_numbers,
            coords,
            to_unit_cell=False,
            coords_are_cartesian=not is_frac,
        )
        ase_atoms = self.adaptor.get_atoms(structure)
        return ase_atoms

    def update_results(
        self, results, predicted_energy, target_energy, predicted_forces, target_force
    ):
        # Update energy predictions and targets
        results["predicted_energy"].append(predicted_energy)
        results["target_energy"].append(target_energy)

        # Update force predictions and targets for each component (fx, fy, fz)
        results["predicted_fx"].extend(predicted_forces[:, 0].tolist())
        results["target_fx"].extend(target_force[:, 0].tolist())
        results["predicted_fy"].extend(predicted_forces[:, 1].tolist())
        results["target_fy"].extend(target_force[:, 1].tolist())
        results["predicted_fz"].extend(predicted_forces[:, 2].tolist())
        results["target_fz"].extend(target_force[:, 2].tolist())

    def save_to_csv(self, data, file_prefix=""):
        os.makedirs(file_prefix, exist_ok=True)
        energy_data = {
            "predicted_energy": data["predicted_energy"],
            "target_energy": data["target_energy"],
        }
        df_energy = pd.DataFrame(energy_data)
        df_energy.to_csv(f"{file_prefix}_energy.csv", index=False)

        # Save force
        force_data = {
            "predicted_fx": data["predicted_fx"],
            "target_fx": data["target_fx"],
            "predicted_fy": data["predicted_fy"],
            "target_fy": data["target_fy"],
            "predicted_fz": data["predicted_fz"],
            "target_fz": data["target_fz"],
        }
        df_force = pd.DataFrame(force_data)
        df_force.to_csv(f"{file_prefix}_force.csv", index=False)


def plot_r2_score(
    test_actual,
    test_pred,
    title="Title",
    suffix="Regular_Mace",
    save_dir="./results/trail_plot/",
):
    save_dir = "./results/trail_plot/"
    os.makedirs(save_dir, exist_ok=True)
    r2_test = r2_score(test_actual, test_pred)
    # Create the scatter plot
    plt.figure(figsize=(6, 6))  # Adjust the figure size as needed
    plt.scatter(test_actual, test_pred, label="Test", color="green")
    plt.xlabel("Target", fontsize=20, fontweight="bold")
    plt.ylabel("Predicted", fontsize=20, fontweight="bold")
    plt.xticks(rotation=90, fontsize=20, fontweight="bold")
    plt.yticks(fontsize=20, fontweight="bold")
    plt.ylim([min(test_actual), max(test_actual)])

    # # Plot the 45-degree line
    # min_val = min(min(test_actual), min(test_pred))
    # max_val = max(max(test_actual), max(test_pred))
    # plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.title(title, fontsize=20, fontweight="bold")

    # Annotate the R² scores on the plot
    plt.text(
        0.05,
        0.90,
        f"Test R² = {r2_test:.5f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        color="green",
    )

    # Save the plot to the specified location with the title in the filename
    filename = f"{title.replace(' ', '_')}_{suffix}.png"
    plt.savefig(f"{save_dir}{filename}", bbox_inches="tight")

    # Show the plot
    plt.show()
    plt.clf()


def main():
    set_random_seeds()
    model_name = "Pretrained-MACE-L0"
    dataset = "MP"
    test_split_path = "/datasets-alt/molecular-data/mat_traj/aug_2024_processed/mptraj-processing/train/"  # "./test_new/test_new"
    batch_size = 1

    predictor = MaterialPropertyPredictor(
        model_name, dataset, test_split_path, batch_size, max_samples=100
    )
    test_loader = predictor.dm.train_dataloader()
    test_results = predictor.process_data_loader(test_loader)
    predictor.save_to_csv(
        test_results, file_prefix=f"./results/test_{model_name}_{dataset}"
    )

    properties = [
        ("energy", "Energy"),
        ("fx", "Fx"),
        ("fy", "Fy"),
        ("fz", "Fz"),
    ]

    for prop, title in properties:
        plot_r2_score(
            test_results[f"target_{prop}"],
            test_results[f"predicted_{prop}"],
            title,
            f"_{model_name}_{dataset}",
        )


if __name__ == "__main__":
    main()
