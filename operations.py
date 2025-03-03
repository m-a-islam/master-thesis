import torch
import torch.nn as nn


OPS = {
    'skip_connect': lambda C, stride: nn.Identity() if stride == 1 else nn.Conv2d(C, C, 1, stride=stride),
    'max_pool_2x2': lambda C, stride: nn.MaxPool2d(2, stride=stride),
    'avg_pool_2x2': lambda C, stride: nn.AvgPool2d(2, stride=stride),
    'conv_1x1': lambda C, stride: nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),
    'conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),
    'conv_5x5': lambda C, stride: nn.Conv2d(C, C, 5, stride=stride, padding=2, bias=False),
    'depthwise_conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=1, groups=C, bias=False),
    'batch_norm': lambda C, stride: nn.BatchNorm2d(C),
}