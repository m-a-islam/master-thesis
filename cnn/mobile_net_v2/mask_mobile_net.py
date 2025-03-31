# MobileNetV2 Architecture
import torch
from torch import nn

from seed_mobile_net import InvertedResidual


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, size_threshold=5.0, mac_threshold=5.0):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)
        self.size_threshold = size_threshold
        self.mac_threshold = mac_threshold
        self.mask = nn.Parameter(torch.ones(7))
        self.block1 = InvertedResidual(32, 16, stride=1, mask=self.mask, mask_index=0)
        self.block2 = InvertedResidual(16, 24, stride=2, mask=self.mask, mask_index=1)
        self.block3 = InvertedResidual(24, 32, stride=2, mask=self.mask, mask_index=2)
        self.block4 = InvertedResidual(32, 64, stride=2, mask=self.mask, mask_index=3)
        self.block5 = InvertedResidual(64, 96, stride=1, mask=self.mask, mask_index=4)
        self.block6 = InvertedResidual(96, 160, stride=2, mask=self.mask, mask_index=5)
        self.block7 = InvertedResidual(160, 320, stride=1, mask=self.mask, mask_index=6)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        """
        # Structural pruning: skip blocks with mask < 0.5
        blocks = [self.block1, self.block2, self.block3, self.block4, self.block5, self.block6, self.block7]
        last_out_channels = 32  # Output from conv1
        for i, block in enumerate(blocks):
            if torch.sigmoid(self.mask[i]) >= 0.5:
                # Adjust input channels if needed (simplified for demo)
                if block.conv1.in_channels != last_out_channels:
                    block.conv1 = nn.Conv2d(last_out_channels, block.conv1.out_channels, kernel_size=1, bias=False)
                x = block(x)
                last_out_channels = block.conv3.out_channels

        # Adjust conv2 input channels to match last block's output
        if last_out_channels != 320:
            self.conv2 = nn.Conv2d(last_out_channels, 1280, kernel_size=1, stride=1, padding=0, bias=False).to(x.device)

        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


    def get_network_description(self):
        """Returns a string describing the active blocks based on masks."""
        description = "Input → Conv1 → BN1 → ReLU → "
        for i in range(7):
            if torch.sigmoid(self.mask[i]) >= 0.5:
                description += f"Block{i+1} → "
        description += "Conv2 → BN2 → ReLU → AvgPool → FC → Output"
        return description
    

    def check_constraints(self, macs, size):
        """Check if the model's MACs and size exceed the defined thresholds, and adjust mask accordingly."""
        if size > self.size_threshold:  # If model size exceeds threshold
            self.apply_pruning_threshold()
        if macs > self.mac_threshold:  # If MACs exceed threshold
            self.apply_pruning_threshold()

    def apply_pruning_threshold(self):
        """Apply pruning based on current thresholds (for both MACs and size)."""
        with torch.no_grad():
            # Reduce the mask (prune) if model size or MACs exceed thresholds
            self.mask.data = self.mask.data * 0.9  # Prune (reduce) the mask values to enforce sparsity