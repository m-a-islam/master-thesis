# MobileNetV2 Architecture
import torch
from torch import nn

from seed_mobile_net import InvertedResidual


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)

        # MobileNetV2 blocks
        self.block1 = InvertedResidual(32, 16, stride=1)
        self.block2 = InvertedResidual(16, 24, stride=2)
        self.block3 = InvertedResidual(24, 32, stride=2)
        self.block4 = InvertedResidual(32, 64, stride=2)
        self.block5 = InvertedResidual(64, 96, stride=1)
        self.block6 = InvertedResidual(96, 160, stride=2)
        self.block7 = InvertedResidual(160, 320, stride=1)

        # Final layers
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

        # Masks for each block, initialized to 1 (will be learned)
        self.mask = nn.Parameter(torch.ones(7))  # 7 layers (block1 to block7)

    def forward(self, x):
        # Apply initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Apply each MobileNetV2 block based on mask values
        mask_1 = torch.sigmoid(self.mask[0])
        mask_2 = torch.sigmoid(self.mask[1])
        mask_3 = torch.sigmoid(self.mask[2])
        mask_4 = torch.sigmoid(self.mask[3])
        mask_5 = torch.sigmoid(self.mask[4])
        mask_6 = torch.sigmoid(self.mask[5])
        mask_7 = torch.sigmoid(self.mask[6])

        # Log initial network before applying masks
        #logger.info("Network before applying mask:")
        #logger.info(str(self))

        if mask_1 > 0.5: x = self.block1(x)
        if mask_2 > 0.5: x = self.block2(x)
        if mask_3 > 0.5: x = self.block3(x)
        if mask_4 > 0.5: x = self.block4(x)
        if mask_5 > 0.5: x = self.block5(x)
        if mask_6 > 0.5: x = self.block6(x)
        if mask_7 > 0.5: x = self.block7(x)

        # Apply the final layers
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # Log network after applying masks
        #logger.info("Network after applying mask:")
        #logger.info(str(self))

        return x