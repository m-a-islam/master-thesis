import torch
import torch.nn.functional as F

from cnn.resNet.resnet_seed import SimpleResNet


class MaskedResNet(SimpleResNet):
    def __init__(self, num_blocks, num_classes=10, mask=None):
        super(MaskedResNet, self).__init__(num_blocks, num_classes)
        self.mask = mask if mask else [True] * 6  # Default to keeping all layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Apply mask on each layer
        if self.mask[0]:
            x = self.layer1(x)
        if self.mask[1]:
            x = self.layer2(x)
        if self.mask[2]:
            x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
