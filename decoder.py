import torch
from torch import nn
from torch.nn import functional as F
from attention import selfAttention


class vaeResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue) ##residual_layer used for same dimensions

class vaeAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = selfAttention(1, channels)
    
    def forward(self, x):
        residue = x     # (batch_size, channel, height, width)
        x = self.groupnorm(x)
        b, c, h, w = x.shape    #(batch_size, channels, height, width)
        x = x.view((b, c, h * w)) #h*w - pixel
        x = x.transpose(-1, -2)     #(batch_size, height*width,  channel)
        x = self.attention(x)
        x = x.transpose(-1,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        -2)       #(batch_size,  channel, height*width)
        x = x.view((b, c, h, w))        #(batch_size, channels, height, width)
        x += residue
        return x 


class vaeDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(8, 8, kernel_size=1, padding=0),      # (batch_size, channel, height/8, width/8)
            nn.Conv2d(8, 512, kernel_size=3, padding=1),
            vaeResidualBlock(512, 512), 
            vaeAttentionBlock(512), 
            vaeResidualBlock(512, 512), 
            vaeResidualBlock(512, 512), 
            vaeResidualBlock(512, 512), 
            vaeResidualBlock(512, 512), 
            
            nn.Upsample(scale_factor=2),         # (batch_size, channel, height/4, width/4) doubles the pixels to increase the size
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            vaeResidualBlock(512, 512), 
            vaeResidualBlock(512, 512), 
            vaeResidualBlock(512, 512), 
        
            nn.Upsample(scale_factor=2),     # (batch_size, channel, height/2, width/2)
    
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            vaeResidualBlock(512, 256), 
            vaeResidualBlock(256, 256), 
            vaeResidualBlock(256, 256), 

            nn.Upsample(scale_factor=2),     # (batch_size, channel, height, width)
        
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            vaeResidualBlock(256, 128), 
            vaeResidualBlock(128, 128), 
            vaeResidualBlock(128, 128), 
            nn.GroupNorm(32, 128), 

            nn.SiLU(), 
            
            nn.Conv2d(128, 3, kernel_size=3, padding=1), # (batch_size, channel, height, width)
        )

    def forward(self, x):

        for module in self:
            x = module(x)
        return x