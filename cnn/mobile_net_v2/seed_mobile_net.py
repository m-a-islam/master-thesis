# MobileNetV2 Block (InvertedResidual)
import torch
from torch import nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6, mask=None, mask_index=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        self.mask = mask  # Learnable mask for this block
        self.mask_index = mask_index

        # Expansion convolution
        self.conv1 = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion)
        self.relu = nn.ReLU6(inplace=True)

        # Depthwise convolution
        self.conv2 = nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, 
                              stride=stride, padding=1, groups=in_channels * expansion, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels * expansion)

        # Pointwise convolution
        self.conv3 = nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x if self.use_res_connect else None
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # Apply mask scaling (differentiable)
        if self.mask is not None and self.mask_index is not None:
            mask_value = torch.sigmoid(self.mask[self.mask_index])
            if self.use_res_connect:
                return identity + mask_value * out
            else:
                return mask_value * out

        else:
            if self.use_res_connect:
                return identity + out
            else:
                return out

