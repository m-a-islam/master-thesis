import torch
import torch.nn as nn

from cnn.resNet.resnet_seed import SimpleResNet

class MaskedResNet(SimpleResNet):
    def __init__(self, num_blocks, num_classes=10, mask=None):
        super(MaskedResNet, self).__init__(num_blocks, num_classes)
        self.mask = nn.Parameter(torch.ones(3))  # Default to keeping all layers
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #debug
        print(f"Before sigmoid - Mask Values: {self.mask.data}")
        maks_1 = self.sigmoid(self.mask[0])
        mask_2 = self.sigmoid(self.mask[1])
        mask_3 = self.sigmoid(self.mask[2])
        print(f"After sigmoid - Mask Values: {maks_1}, {mask_2}, {mask_3}")

        # Apply mask on each layer
        if maks_1 > 0.5:
            x = self.layer1(x)
        else:
            x = x.clone()  # Ensure the layer is called

        if mask_2 > 0.5:
            x = self.layer2(x)
        else:
            x = x.clone()  # Ensure the layer is called

        if mask_3 > 0.5:
            x = self.layer3(x)
        else:
            x = x.clone()  # Ensure the layer is called

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x