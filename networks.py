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
            nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False),
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
            nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False),
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

        # Encoder
        self.encoder_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(net_in_channels, encoder_filters[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )])

        for i in range(len(encoder_filters)-2):
            self.encoder_layers.append(
                EncoderBlock(encoder_filters[i], encoder_filters[i+1])
            )

        # Note: See Errata
        self.encoder_layers.append(nn.Sequential(
            nn.Conv2d(encoder_filters[-2], encoder_filters[-1], 4, 2, 1),
            nn.ReLU()
        ))

        # Decoder
        self.decoder_layers = nn.ModuleList([DecoderBlock(encoder_filters[-1], decoder_filters[0])])

        for i in range(len(decoder_filters)-1):
            self.decoder_layers.append(DecoderBlock(2*decoder_filters[i], decoder_filters[i+1]))

        self.decoder_layers.append(nn.Sequential(
            nn.ConvTranspose2d(2*decoder_filters[-1], net_out_channels, 4,2,1),
            nn.Tanh()
        ))

    def forward(self, X):
        enc_activations = [X]
        for enc in self.encoder_layers:
            enc_activations.append(enc(enc_activations[-1]))

        activation = self.decoder_layers[0](enc_activations[-1])
        
        for i in range(1, len(self.decoder_layers)):
            activation = torch.cat([activation, enc_activations[-(i+1)]], dim = 1)
            activation = self.decoder_layers[i](activation)

        return activation