# model_search.py
# Basic search model for CNN operations

import torch
import torch.nn as nn
from phylum import SEARCH_SPACE
from operations import OPS

class MixedOp(nn.Module):
    """
    A 'mixed' operation representing one choice among multiple
    CNN operations, weighted by architecture parameters.
    """
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in SEARCH_SPACE.get_operations('CNN'):
            op = OPS[primitive](C, stride)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """
        x: input tensor
        weights: architecture parameters for each op
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))
