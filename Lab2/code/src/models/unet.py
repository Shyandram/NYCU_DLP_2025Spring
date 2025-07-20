# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x_down = self.pool(x)
        return x, x_down

class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, upsample='bilinear'):
        super(decoder_block, self).__init__()
        if upsample == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif upsample == 'Deconv':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels//2, in_channels//2, 2, stride=2),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError("Invalid value for 'upsample'. Use 'bilinear' or 'Deconv'")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x_cat = torch.cat([x1, x2], dim=1)
        x = F.relu(self.conv1(x_cat))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_channels=64, upsample='bilinear'):  
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, n_channels, 3, stride=1, padding=1)
        self.down1 = encoder_block(n_channels, n_channels) # 64
        self.down2 = encoder_block(n_channels, n_channels*2) # 128
        self.down3 = encoder_block(n_channels*2, n_channels*4) # 256
        self.down4 = encoder_block(n_channels*4, n_channels*8) # 512

        self.mid = nn.Sequential(
            nn.Conv2d(n_channels*8, n_channels*16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels*16, n_channels*8, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = decoder_block(n_channels*16, n_channels*4, upsample=upsample) # 512
        self.up2 = decoder_block(n_channels*8, n_channels*2, upsample=upsample) # 256
        self.up3 = decoder_block(n_channels*4, n_channels, upsample=upsample)
        self.up4 = decoder_block(n_channels*2, n_channels, upsample=upsample)
        self.out = nn.Sequential(
            nn.Conv2d(n_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        assert x.shape[1] == self.in_channels

        x = F.relu(self.conv1(x))
        x1, x1_down = self.down1(x)
        x2, x2_down = self.down2(x1_down)
        x3, x3_down = self.down3(x2_down)
        x4, x4_down = self.down4(x3_down)
        
        x = self.mid(x4_down) # 512x32x32

        x = self.up1(x, x4) # 256x64x64
        x = self.up2(x, x3) # 128x128x128
        x = self.up3(x, x2) # 64x256x256
        x = self.up4(x, x1) # 64x512x512
        x = self.out(x)
        return x


if __name__ == '__main__':
    # Test your network
    model = UNet(3, upsample='Deconv')
    x = torch.randn(1, 3, 512, 512)
    print(model(x).shape)