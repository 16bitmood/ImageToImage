from turtle import forward
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, 
                stride=stride, padding=1, bias=False, padding_mode='reflect'
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, X):
        return self.model(X)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, dropout_rate=None):
        super().__init__()
        layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, 
                stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ])
        if dropout_rate is not None:
            layers.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

class PatchDiscriminator(nn.Module):
    def __init__(self, net_in_channels, filters=[64, 128, 256, 512]):
        # C64-C128-C256-C512
        super().__init__()
        # Notes:
        # - We get Two Images Stacked on last axis, so first in_channels = in_channels*2

        initial = nn.Sequential(
            nn.Conv2d(net_in_channels*2, filters[0], 4, 2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        layers = nn.ModuleList([initial])
        in_channels = filters[0]

        for out_channels in filters[1:]:
            layers.append(
                EncoderBlock(in_channels, out_channels, stride=(1 if out_channels == filters[-1] else 2))
            )
            in_channels = out_channels
        
        layers.append(
            nn.Conv2d(
                in_channels, out_channels=1, kernel_size=4,
                stride=1, padding=1, padding_mode='reflect'
            )
        )

        self.model = nn.Sequential(*layers)
        
    def forward(self, X, Y):
        XY = torch.cat([X, Y], dim = 1)
        return self.model(XY)

class UNetGenerator(nn.Module):
    def __init__(
            self, 
            net_in_channels, 
            net_out_channels, 
            encoder_filters = [64, 128, 256, 512, 512, 512, 512, 512],
            decoder_filters = [512, 512, 512, 512, 256, 128, 64]):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(net_in_channels, encoder_filters[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.enc2 = EncoderBlock(encoder_filters[0], encoder_filters[1])
        self.enc3 = EncoderBlock(encoder_filters[1], encoder_filters[2])
        self.enc4 = EncoderBlock(encoder_filters[2], encoder_filters[3])
        self.enc5 = EncoderBlock(encoder_filters[3], encoder_filters[4])
        self.enc6 = EncoderBlock(encoder_filters[4], encoder_filters[5])
        self.enc7 = EncoderBlock(encoder_filters[5], encoder_filters[6])

        # See Errata
        self.enc8 = nn.Sequential(
            nn.Conv2d(encoder_filters[6], encoder_filters[7], 4, 2, 1),
            nn.ReLU()
        )

        self.dec1 = DecoderBlock(encoder_filters[-1], decoder_filters[0])
        self.dec2 = DecoderBlock(2 * decoder_filters[0], decoder_filters[1])
        self.dec3 = DecoderBlock(2 * decoder_filters[1], decoder_filters[2])
        self.dec4 = DecoderBlock(2 * decoder_filters[2], decoder_filters[3])
        self.dec5 = DecoderBlock(2 * decoder_filters[3], decoder_filters[4])
        self.dec6 = DecoderBlock(2 * decoder_filters[4], decoder_filters[5])
        self.dec7 = DecoderBlock(2 * decoder_filters[5], decoder_filters[6])

        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(2 * decoder_filters[6], net_out_channels, 4,2,1),
            nn.Tanh()
        )

    def forward(self, X):
        E1 = self.enc1(X)
        E2 = self.enc2(E1)
        E3 = self.enc3(E2)
        E4 = self.enc4(E3)
        E5 = self.enc5(E4)
        E6 = self.enc6(E5)
        E7 = self.enc7(E6)

        E8 = self.enc8(E7)
        D1 = self.dec1(E8)

        D2 = self.dec2(torch.cat([D1, E7], dim = 1))
        D3 = self.dec3(torch.cat([D2, E6], dim = 1))
        D4 = self.dec4(torch.cat([D3, E5], dim = 1))
        D5 = self.dec5(torch.cat([D4, E4], dim = 1))
        D6 = self.dec6(torch.cat([D5, E3], dim = 1))
        D7 = self.dec7(torch.cat([D6, E2], dim = 1))
        D8 = self.dec8(torch.cat([D7, E1], dim = 1))

        return D8