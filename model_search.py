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
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS:
            op = OPS[primitive](C, stride)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        assert len(weights) == len(self._ops), "Weights and operations length mismatch"
        result = 0
        for w, op in zip(weights, self._ops):
            out = op(x)
            if out.shape != x.shape:
                # Adjust the output shape to match the input shape
                out = nn.functional.interpolate(out, size=x.shape[2:])
            result += w * out
        return result