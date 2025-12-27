# U-Net implementation  in PyTorch
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),

            #the input of this layer should be the output of the previous layer
            #I changed the input
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # encoding phase
        # as depth increases, number of feature channels increases
        # patial resolution decreases
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        
        # downsampling with maxpooling
        self.pool = nn.MaxPool2d(kernel_size=2)

        # upsampling 
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # phase of decoding
        self.dec2 = ConvBlock(128 + 256, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        # coverting the feature vector to 
        # the class scores logits
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)



    def forward(self, x):
        # encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # decoding path

        d2 = self.up(e3)
        # skipping connections
        d2 = self.dec2(torch.cat([d2, e2], dim =1))

        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1,e1], dim =1))

        return self.final(d1)