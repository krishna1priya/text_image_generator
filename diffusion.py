import torch
from torch import nn
from torch.nn import functional as F
from attention import selfAttention, crossAttention

class timeEmbedding(nn.Module):             #encodes info in the present time step 
    def __init__(self, num_embed):
        super().__init__()
        self.linear_1 = nn.Linear(num_embed, 4 * num_embed)
        self.linear_2 = nn.Linear(4 * num_embed, 4 * num_embed)

    def forward(self, x):           # x -  (1, 320)
        x = self.linear_1(x)        #(1, 1280)
        x = F.silu(x)               #(1, 1280)
        x = self.linear_2(x)        #(1, 1280)
        return x

class unetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupNorm = nn.GroupNorm(32, in_channels)
        self.conV = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupNorm2 = nn.GroupNorm(32, out_channels)
        self.conV2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()         
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):           #relating latent to time embed
        #input: feature - latent, time - time embedding
        residue = feature
        feature = self.groupNorm(feature)
        feature = F.silu(feature)
        feature = self.conV(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)         # # (batch_size, channels, height, width) + (1, channels, 1, 1)
        merged = self.groupNorm2(merged)
        merged = F.silu(merged)
        merged = self.conV2(merged)
        return merged + self.residual_layer(residue)

class unetAttentionBlock(nn.Module):
    def __init__(self, num_heads: int, num_embed: int, d_prompt=768):
        super().__init__()
        channels = num_heads * num_embed
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = selfAttention(num_heads, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = crossAttention(num_heads, channels, d_prompt, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_gelu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_gelu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, prompt):
        #input: feature - latent, prompt - (batch_size, seq_length, dim=768)
        residue_end = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        b, c, h, w = x.shape  #batch_size, channels, height, width
        x = x.view((b, c, h * w))      # (batch_size, channels, height*width)
        x = x.transpose(-1, -2)          # (batch_size, height*width, channels)
        
        # Normalization + Self-Attention with skip connection
        residue_start = x           #to be applied after attention
        x = self.layernorm_1(x)
        x = self.attention_1(x)            # (batch_size, height*width, channels)
        x += residue_start
    
        residue_start = x

        # Normalization + Cross-Attention with skip connection
        x = self.layernorm_2(x)
        x = self.attention_2(x, prompt)
        x += residue_start
        
        residue_start = x

        x = self.layernorm_3(x)
        x, gate = self.linear_gelu_1(x).chunk(2, dim=-1)        # (batch_size, height*width, channels*4)
        x = x * F.gelu(gate)
        x = self.linear_gelu_2(x)
        x += residue_start
        
        x = x.transpose(-1, -2)     # (batch_size, channels, height*width)

        x = x.view((b, c, h, w))         # (batch_size, channels, height, width)
        return self.conv_output(x) + residue_end

class upSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')        # (batch_size, channels, height*2, width*2)
        return self.conv(x)

class switchSequential(nn.Sequential):
    def forward(self, x, prompt, time):
        for layer in self:
            if isinstance(layer, unetAttentionBlock):          #computes cross attention bw latent and prompt
                x = layer(x, prompt)
            elif isinstance(layer, unetResidualBlock):         #matches latent with time step
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            switchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),                  # (batch_size, 320, height/8, width/8)
            switchSequential(unetResidualBlock(320, 320), unetAttentionBlock(8, 40)),       # (batch_size, 320, height/8, width/8)
            switchSequential(unetResidualBlock(320, 320), unetAttentionBlock(8, 40)),        # (batch_size, 320, height/8, width/8)
            switchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),       # (batch_size, 320, height/16, width/16)
            switchSequential(unetResidualBlock(320, 640), unetAttentionBlock(8, 80)),        # (batch_size, 640, height/16, width/16)
            switchSequential(unetResidualBlock(640, 640), unetAttentionBlock(8, 80)),        # (batch_size, 640, height/16, width/16)
            switchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),      # (batch_size, 640, height/32, width/32)
            switchSequential(unetResidualBlock(640, 1280), unetAttentionBlock(8, 160)),     # (batch_size, 1280, height/32, width/32)
            switchSequential(unetResidualBlock(1280, 1280), unetAttentionBlock(8, 160)),        # (batch_size, 1280, height/32, width/32)
            switchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),         # (batch_size, 1280, height/64, width/64)
            switchSequential(unetResidualBlock(1280, 1280)),                                     # (batch_size, 1280, height/64, width/64)
            switchSequential(unetResidualBlock(1280, 1280)),                                 # (batch_size, 1280, height/64, width/64)
        ])

        self.bottleneck = switchSequential(
            unetResidualBlock(1280, 1280),   # (batch_size, 1280, height/64, width/64)
            unetAttentionBlock(8, 160), 
            unetResidualBlock(1280, 1280), 
        )
        
        self.decoders = nn.ModuleList([
            switchSequential(unetResidualBlock(2560, 1280)),           ##2560 - as skip connections doubles the size -   # (batch_size, 2560, height/64, width/64) to   # (batch_size, 1280, height/64, width/64)
            switchSequential(unetResidualBlock(2560, 1280)),
            switchSequential(unetResidualBlock(2560, 1280), upSample(1280)),         # (batch_size, 1280, height/32, width/32)
            switchSequential(unetResidualBlock(2560, 1280), unetAttentionBlock(8, 160)),
            switchSequential(unetResidualBlock(2560, 1280), unetAttentionBlock(8, 160)),
            switchSequential(unetResidualBlock(1920, 1280), unetAttentionBlock(8, 160), upSample(1280)),        # (batch_size, 1280, height/16, width/16)
            switchSequential(unetResidualBlock(1920, 640), unetAttentionBlock(8, 80)),
            switchSequential(unetResidualBlock(1280, 640), unetAttentionBlock(8, 80)),
            switchSequential(unetResidualBlock(960, 640), unetAttentionBlock(8, 80), upSample(640)),             # (batch_size, 1280, height/8, width/8)
            switchSequential(unetResidualBlock(960, 320), unetAttentionBlock(8, 40)),
            switchSequential(unetResidualBlock(640, 320), unetAttentionBlock(8, 40)),
            switchSequential(unetResidualBlock(640, 320), unetAttentionBlock(8, 40)),
        ])

    def forward(self, x, prompt, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # prompt: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, prompt, time)
            skip_connections.append(x)

        x = self.bottleneck(x, prompt, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, prompt, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.groupnorm(x)             # (batch_size, 320, height/8, width/8)
        x = F.silu(x)
        x = self.conv(x)             # (batch_size, 4, height/8, width/8)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = timeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, prompt, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)        # output of Variational Encoder
        # prompt: (Batch_Size, Seq_Len, Dim)                   #text prompt - output from CLIP
        # time: (1, 320)                                        #time of the latent getting noisified
        time = self.time_embedding(time)                # (1, 1280 = 4*num_embed)  #arrival step in denoisification
    
        output = self.unet(latent, prompt, time)       #(Batch, 320, Height / 8, Width / 8)        #predicts noise and removes it
        output = self.final(output) #(Batch, 4, Height / 8, Width / 8)
        return output