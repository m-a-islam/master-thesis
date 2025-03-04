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
            # Add BatchNorm after pooling if stride=1 to match DARTS behavior
            if 'pool' in primitive and stride == 1:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """
        x: input tensor
        weights: architecture parameters for each op
        """
        if weights.dim() == 0:
            weights = weights.unsqueeze(0)
        return sum(w * op(x) for w, op in zip(weights, self._ops))