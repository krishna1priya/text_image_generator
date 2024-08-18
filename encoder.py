import torch
from torch import nn
from torch.nn import functional as F
from decoder import vaeAttentionBlock, vaeResidualBlock

class vaeEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), # (batch_size, channel, height, width), channels = 128
            vaeResidualBlock(128, 128),
            vaeResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), # (batch_size, channel, height/2, width/2)
            vaeResidualBlock(128, 256),   ##increasing features
            vaeResidualBlock(256, 256), 
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),    # (batch_size, channel, height/4, width/4)
            vaeResidualBlock(256, 512), ##increasing features
            vaeResidualBlock(512, 512), 
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), # (batch_size, channel, height/8, width/8)
            vaeResidualBlock(512, 512), 
            vaeResidualBlock(512, 512), 
            vaeResidualBlock(512, 512), 
            vaeAttentionBlock(512), 
            vaeResidualBlock(512, 512), 
            nn.GroupNorm(32, 512), ##group normalization, #groups = 32, #channels = features = 512
            nn.SiLU(), 
            nn.Conv2d(512, 8, kernel_size=3, padding=1),  #decreasing features
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x, noise):
        # x: (batch_size, channel, height, width) h = 512, w = 512
        # noise: (batch_size, channel, height/8, width/8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2): ##assymtric padding to right and bottom
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -20, 20) #clamping to a range
        variance = log_variance.exp()
        stdev = variance.sqrt()

        x = mean + stdev * noise
        
        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        # x *= 0.18215
        
        return x

