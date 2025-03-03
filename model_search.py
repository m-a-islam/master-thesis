# model_search.py
# Basic search model for CNN operations

import torch
import torch.nn as nn
from phylum import SEARCH_SPACE
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
        """
        x: input tensor
        weights: architecture parameters for each op
        """
        if weights.dim() == 0:
            weights = weights.unsqueeze(0)
        return sum(w * op(x) for w, op in zip(weights, self._ops))