from pathlib import Path
from types import MethodType
from typing import Any, Union

import dgl
import matgl
import torch

# fmt: off
atomic_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
 89, 90, 91, 92, 93, 94]
# fmt: on


def forward(
    self,
    batch: dict[str, Any],
    state_attr: Union[torch.Tensor, None] = None,
    l_g: Union[dgl.DGLGraph, None] = None,
):
    lat = batch["cell"]
    graph = batch["graph"]
    atomic_numbers = graph.ndata["node_type"]
    graph.ndata["node_type"] = torch.Tensor(
        [self.atomic_number_map[num.item()] for num in atomic_numbers]
    ).long()
    total_energies, forces, stresses, *others = self.matgl_forward(
        graph, lat, state_attr, l_g
    )
    output = {}
    output["energy"] = total_energies
    output["forces"] = forces
    output["stress"] = stresses
    return output


def load_matgl(ckpt):
    matgl.float_th = torch.float64
    matgl_model = matgl.load_model(Path(ckpt).parent)
    matgl_model.matgl_forward = matgl_model.forward
    matgl_model.forward = MethodType(forward, matgl_model)
    matgl_model.predict = MethodType(forward, matgl_model)
    matgl_model.atomic_number_map = dict(
        zip(atomic_numbers, range(len(atomic_numbers)))
    )
    return matgl_model
