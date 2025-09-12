import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

import torchsummary

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, skip):
        # x is from previous layer, skip is from encoder
        # x (B, in_channels, H, W)
        # skip (B, in_channels//2, H*2, W*2)
        x = self.up(x)

        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        assert x.size()[1] == skip.size()[1] and x.size()[2] == skip.size()[2] and x.size()[3] == skip.size()[3], "Skip connection and upsampled feature map sizes do not match"

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=40, init_features=64):
        super(UNet, self).__init__()
        features = init_features

        # Encoder
        self.e1 = DoubleConv(in_channels, features, features)
        self.e2 = Down(features, features * 2)
        self.e3 = Down(features * 2, features * 4)
        self.e4 = Down(features * 4, features * 8)

        self.encoder = nn.Sequential(OrderedDict([
            ('enc1', self.e1),
            ('enc2', self.e2),
            ('enc3', self.e3),
            ('enc4', self.e4),
        ]))

        self.bottleneck = Down(features * 8, features * 16)

        # Decoder
        self.u4 = Up(features * 16, features * 8)
        self.u3 = Up(features * 8,  features * 4)
        self.u2 = Up(features * 4,  features * 2)
        self.u1 = Up(features * 2,  features)

        self.decoder = nn.Sequential(OrderedDict([
            ('dec4', self.u4),
            ('dec3', self.u3),
            ('dec2', self.u2),
            ('dec1', self.u1),
        ]))

        # Final Convolution
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        encoders = []
        for name, layer in self.encoder.named_children():
            x = layer(x)
            encoders.append(x)

        x = self.bottleneck(x)
        encoders = encoders[::-1]

        for i, (name, layer) in enumerate(self.decoder.named_children()):
            x = layer(x, encoders[i])

        last_conv = self.conv(x)
        return last_conv
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = UNet(in_channels=3, out_channels=39, init_features=32).to(device)

    torchsummary.summary(model, (3, 128, 256), verbose=2, depth=1)

    x = torch.randn((1, 3, 128, 256)).to(device)

    logits = model(x)
    print(logits.shape)