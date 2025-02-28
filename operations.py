import torch
import torch.nn as nn

def create_mlp(C):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(C * 7 * 7, C),
        nn.ReLU(),
        nn.Linear(C, C)
    )

def create_se_block(C):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(C, C // 16, 1),
        nn.ReLU(),
        nn.Conv2d(C // 16, C, 1),
        nn.Sigmoid()
    )

OPS = {
    'none': lambda C, stride: nn.Identity(),
    'skip_connect': lambda C, stride: nn.Identity() if stride == 1 else nn.Conv2d(C, C, 1, stride=stride),
    'max_pool_2x2': lambda C, stride: nn.MaxPool2d(2, stride=stride),
    'avg_pool_2x2': lambda C, stride: nn.AvgPool2d(2, stride=stride),
    'dropout': lambda C, stride: nn.Dropout(p=0.3),
    'conv_1x1': lambda C, stride: nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),
    'conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),
    'conv_5x5': lambda C, stride: nn.Conv2d(C, C, 5, stride=stride, padding=2, bias=False),
    'conv_7x7': lambda C, stride: nn.Conv2d(C, C, 7, stride=stride, padding=3, bias=False),
    'depthwise_conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=1, groups=C, bias=False),
    'dilated_conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=2, dilation=2, bias=False),
    'grouped_conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=1, groups=2, bias=False),
    'batch_norm': lambda C, stride: nn.BatchNorm2d(C),
    'layer_norm': lambda C, stride: nn.LayerNorm([C, 7, 7]),
    'mlp': lambda C, stride: create_mlp(C),
    'squeeze_excitation': lambda C, stride: create_se_block(C),
    'max_pool_3x3': lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    'avg_pool_3x3': lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1),
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
    'dil_conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=2, dilation=2, bias=False),
    'dil_conv_5x5': lambda C, stride: nn.Conv2d(C, C, 5, stride=stride, padding=4, dilation=2, bias=False),
}