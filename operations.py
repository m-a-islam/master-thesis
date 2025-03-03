import torch
import torch.nn as nn

# operations.py
OPS = {
    'none': lambda C, stride: nn.Identity(),
    'skip_connect': lambda C, stride: nn.Identity() if stride == 1 else nn.Conv2d(C, C, 1, stride=stride),
    'max_pool_2x2': lambda C, stride: nn.MaxPool2d(2, stride=stride),
    'avg_pool_2x2': lambda C, stride: nn.AvgPool2d(2, stride=stride),
    'conv_1x1': lambda C, stride: nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),
    'conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),
    'conv_5x5': lambda C, stride: nn.Conv2d(C, C, 5, stride=stride, padding=2, bias=False),
    'sep_conv_3x3': lambda C, stride: nn.Sequential(
        nn.Conv2d(C, C, 3, stride=stride, padding=1, groups=C, bias=False),
        nn.Conv2d(C, C, 1, bias=False),
        nn.BatchNorm2d(C),
        nn.ReLU(inplace=True)
    ),
    'sep_conv_5x5': lambda C, stride: nn.Sequential(
        nn.Conv2d(C, C, 5, stride=stride, padding=2, groups=C, bias=False),
        nn.Conv2d(C, C, 1, bias=False),
        nn.BatchNorm2d(C),
        nn.ReLU(inplace=True)
    ),
    'sep_conv_7x7': lambda C, stride: nn.Sequential(
        nn.Conv2d(C, C, 7, stride=stride, padding=3, groups=C, bias=False),
        nn.Conv2d(C, C, 1, bias=False),
        nn.BatchNorm2d(C),
        nn.ReLU(inplace=True)
    ),
    'dil_conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=2, dilation=2, bias=False),
    'dil_conv_5x5': lambda C, stride: nn.Conv2d(C, C, 5, stride=stride, padding=4, dilation=2, bias=False),
    'conv_7x1_1x7': lambda C, stride: nn.Sequential(
        nn.Conv2d(C, C, (7, 1), stride=stride, padding=(3, 0), bias=False),
        nn.Conv2d(C, C, (1, 7), stride=stride, padding=(0, 3), bias=False),
        nn.BatchNorm2d(C),
        nn.ReLU(inplace=True)
    )
}
