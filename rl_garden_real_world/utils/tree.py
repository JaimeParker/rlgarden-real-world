"""Small pytree helpers shared by teleop recording and data-collection tools."""
from __future__ import annotations

import numpy as np


def index_tree(tree, index: int):
    import torch

    if isinstance(tree, dict):
        return {key: index_tree(value, index) for key, value in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree[index].detach().cpu()
    return np.asarray(tree[index])
