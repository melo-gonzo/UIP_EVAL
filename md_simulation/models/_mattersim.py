import torch
from types import MethodType
from ase.units import GPa
from mattersim.forcefield.potential import Potential
from mattersim.datasets.utils.build import build_dataloader


def load_pretrained_mattersim():
    mattersim_model = Potential.from_checkpoint(
        load_path="mattersim-v1.0.0-5m", device="cpu"
    )

    def forward(self, atoms):
        dataloader = build_dataloader([atoms], only_inference=True)
        predictions = mattersim_model.predict_properties(
            dataloader, include_forces=True, include_stresses=True
        )
        s = predictions[2] * GPa  # eV/A^3
        stress = torch.tensor([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        results = {"energy": predictions[0], "forces": predictions[1], "stress": stress}
        return results

    mattersim_model.mace_forward = mattersim_model.forward
    mattersim_model.forward = MethodType(forward, mattersim_model)
    return mattersim_model


# # set up the structure
# si = bulk("Si", "diamond", a=5.43)

# # replicate the structures to form a list
# structures = [si] * 10

# # load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Running MatterSim on {device}")
# potential = Potential.from_checkpoint(device=device)

# # build the dataloader that is compatible with MatterSim
# dataloader = build_dataloader(structures, only_inference=True)

# # make predictions
# predictions = potential.predict_properties(dataloader, include_forces=True, include_stresses=True)

# # print the predictions
# print(f"Total energy in eV: {predictions[0]}")
# print(f"Forces in eV/Angstrom: {predictions[1]}")
# print(f"Stresses in GPa: {predictions[2]}")
# print(f"Stresses in eV/A^3: {np.array(predictions[2])*GPa}")
