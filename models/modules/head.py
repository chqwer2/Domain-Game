from torch import nn
from .act import Activation

def extra_stem(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.LayerNorm(out_channel)
    )

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),  # SiLU
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=kernel_size // 2),
        )
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) \
                        if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


import torch

class SCSEmask(nn.Module):
    def __init__(self, in_channels, reduction=2, with_weight=True):
        super().__init__()
        self.with_weight = with_weight
        
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        # self.mask
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), # in, out, channel size
                                 nn.Sigmoid())
        
        self.mask = nn.Sequential(nn.Conv2d(in_channels, in_channels, 7, padding=3), 
                                  nn.Softmax(dim = 1))
        
        self.printability = 1
        
    def forward(self, x):
        if self.printability:
            print("x in attention", x.shape)
        
        # s = x.shape
        x = x * self.cSE(x) + x * self.sSE(x)
        
        w = self.mask(x)
        w = (x == torch.max(x, dim=1, keepdim=True).values) * 1.0
        
        if self.printability:
            print("x, w in printability", x.shape, w.shape)
            self.printability = 0
            
        return w, x 
    

    

class ReconstructionHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        super().__init__()
                
        slice = 4
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, slice * out_channels,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),  # SiLU
            nn.Conv2d(slice * out_channels, slice * out_channels,
                      kernel_size=kernel_size, padding=kernel_size // 2),
        )
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) \
                          if upsampling > 1 else nn.Identity()
        
        
        self.activation = Activation(activation)
        self.softmax = nn.Softmax(dim=1)
        
        self.components_attention = SCSEmask(4, with_weight=True)
        
        
    def forward(self, x, return_comp=False):
        
        x = self.conv2d(x)
        x = self.upsampling(x)
        x = self.activation(x)
        
        # softscore = x * self.softmax(x)
        
        score, x = self.components_attention(x)
        
        if return_comp:
            # recon, comp
            return (score * x).sum(dim=1, keepdim=True), x
        
        return (score * x).sum(dim=1, keepdim=True)
    


