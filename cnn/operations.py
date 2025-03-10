import torch.nn as nn

OPS = {
    'conv_3x3': lambda C, stride: nn.Sequential(
        nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(C)
    ),
    'conv_1x1': lambda C, stride: nn.Sequential(
        nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(C)
    ),
    'max_pool_3x3': lambda C, stride: nn.Sequential(
        nn.MaxPool2d(3, stride=stride, padding=1),
        nn.BatchNorm2d(C)
    ),
    'avg_pool_3x3': lambda C, stride: nn.Sequential(
        nn.AvgPool2d(3, stride=stride, padding=1),
        nn.BatchNorm2d(C)
    ),
    'skip_connect': lambda C, stride: nn.Identity() if stride == 1 else nn.Sequential(nn.AvgPool2d(2, stride=2))  # Handle strided skip
}