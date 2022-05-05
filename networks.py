import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kind, act, drop=None, **kwargs):
        super().__init__()
        Conv = nn.Conv2d if kind=='down' else nn.ConvTranspose2d 

        self.block = [Conv(in_c, out_c, **kwargs)]
        self.block.append(
            nn.InstanceNorm2d(out_c))

        if act == 'relu':
            self.block.append(nn.ReLU(True))
        elif act == 'leaky':
            self.block.append(nn.LeakyReLU(0.2, True))
        elif act != 'none':
            raise Exception(f'Invalid Activation')

        if drop is not None:
            self.block.append(nn.Dropout(drop))

        self.block = nn.Sequential(*self.block)

    def forward(self, X):
        return self.block(X)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.block = ConvBlock(
            in_channels, out_channels, kind='down', act='leaky',
            kernel_size=4,  stride=stride, padding=1, bias=False, padding_mode='reflect')

    def forward(self, X):
        return self.block(X)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, dropout_rate=None):
        super().__init__()
        self.block = ConvBlock(
            in_channels, out_channels, kind='up', act='relu', drop=0.5,
            kernel_size=4,  stride=stride, padding=1, bias=False)

    def forward(self, X):
        return self.block(X)

class PatchDiscriminator(nn.Module):
    def __init__(self, net_in_channels, conditional=True, filters=[64, 128, 256, 512]):
        super().__init__()
        # Notes:
        # - For conditional, We get 2 images stacked on last axis
        #   so first in_channels = in_channels*2
        
        self.conditional = conditional
        net_in_channels = net_in_channels*2 if conditional else net_in_channels
        initial = nn.Sequential(
            nn.Conv2d(net_in_channels, filters[0], 4, 2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
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

        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, X, Y=None):
        if self.conditional:
            XY = torch.cat([X, Y], dim = 1)
            return self.model(XY)
        else:
            return self.model(X)

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

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kind='down', act='relu', kernel_size=3, padding=1),
            ConvBlock(channels, channels, kind='down', act='none', kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, net_in_channels, net_out_channels, filters = [64, 128, 256], num_residuals=9):
        super().__init__()

        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(nn.Sequential(
            nn.Conv2d(net_in_channels, filters[0], kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        ))

        for i in range(len(filters)-1):
            self.encoder_layers.append(ConvBlock( 
                filters[i], filters[i+1], kind='down', act='relu', kernel_size=3, stride=2, padding=1))
        
        self.residual_layers = nn.Sequential(*[ResNetBlock(filters[-1]) for _ in range(num_residuals)])

        self.decoder_layers = nn.ModuleList()
        for i in range(len(filters)-1):
            self.decoder_layers.append(ConvBlock(
                filters[-(i+1)], filters[-(i+2)], kind='up', act='relu', kernel_size=3, stride=2, padding=1, output_padding=1))

        self.decoder_layers.append(nn.Conv2d(
            filters[0], net_out_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect"))

        self.decoder_layers.append(nn.Tanh())

    def forward(self, X):
        for layer in self.encoder_layers:
            X = layer(X)
        for layer in self.residual_layers:
            X = layer(X)
        for layer in self.decoder_layers:
            X = layer(X)

        return X