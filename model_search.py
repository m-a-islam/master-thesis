# model_search.py
# Basic search model for CNN operations
# notes: is the weighted sum of all operations that can be applied to the input edge,
# where the weights are the learned during the optimization process.
# The operations are defined in the operations.py file.

import torch
import torch.nn as nn
from operations import OPS

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS:
            op = OPS[primitive](C, stride)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))