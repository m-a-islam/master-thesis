# MobileNetV2 Block (InvertedResidual)
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        # First convolution (1x1 expansion)
        self.conv1 = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion)
        self.relu = nn.ReLU6(inplace=True)

        # Depthwise convolution (3x3)
        self.conv2 = nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels * expansion)

        # Last convolution (1x1 pointwise linear)
        self.conv3 = nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.use_res_connect:
            x += identity
        return x
