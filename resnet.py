# ResNet generator and discriminator
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm
import numpy as np


channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1, bias=False)
        nn.init.xavier_normal_(self.conv1.weight.data, 0.02)
        nn.init.xavier_normal_(self.conv2.weight.data, 0.02)
        nn.init.xavier_normal_(self.upsample.weight.data, 0.02)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            self.upsample,
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            nn.init.xavier_normal_(self.bypass.weight.data, 0.02)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.pooling1 = nn.Conv2d(out_channels, out_channels, 4, 2, 1, groups=out_channels, bias=False)
        nn.init.xavier_normal_(self.conv1.weight.data, 0.02)
        nn.init.xavier_normal_(self.conv2.weight.data, 0.02)
        nn.init.xavier_normal_(self.pooling1.weight.data, 0.02)

        if stride == 1:
            self.model = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(self.conv1),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(self.conv1),
                nn.LeakyReLU(0.2, inplace=True),
                SpectralNorm(self.conv2),
                self.pooling1
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            self.pooling2 = nn.Conv2d(out_channels, out_channels, 4, 2, 1, groups=out_channels, bias=False)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
            nn.init.xavier_normal_(self.pooling2.weight.data, 0.02)

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                self.pooling2
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.pooling1 = nn.Conv2d(out_channels, out_channels, 4, 2, 1, groups=out_channels, bias=False) 
        self.pooling2 = nn.Conv2d(in_channels, in_channels, 4, 2, 1, groups=in_channels, bias=False)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_normal_(self.conv1.weight.data, 0.02)
        nn.init.xavier_normal_(self.conv2.weight.data, 0.02)
        nn.init.xavier_normal_(self.pooling1.weight.data, 0.02)
        nn.init.xavier_normal_(self.pooling2.weight.data, 0.02)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(self.conv2),
            self.pooling1
            )
        self.bypass = nn.Sequential(
            self.pooling2,
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=128
DISC_SIZE=128

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.ConvTranspose2d(self.z_dim, GEN_SIZE, 4, 1, 0, bias=False)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_normal_(self.dense.weight.data, 0.02)
        nn.init.xavier_normal_(self.final.weight.data, 0.02)

        self.model = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.pooling8 = nn.Conv2d(DISC_SIZE, DISC_SIZE, 8, 4, 1, groups=DISC_SIZE, bias=False)
        nn.init.xavier_normal_(self.pooling8.weight.data, 0.02)
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.LeakyReLU(0.2, inplace=True),
                self.pooling8,
            )
        self.fc = nn.Conv2d(DISC_SIZE, 1, 1, bias=False)
        nn.init.xavier_normal_(self.fc.weight.data, 0.02)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x)).squeeze()