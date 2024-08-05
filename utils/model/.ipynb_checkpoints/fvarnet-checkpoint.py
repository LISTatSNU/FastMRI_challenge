import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
from varnet import SensitivityModel, VarNetBlock, NormUnet
from unet import Unet

def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w, two = x.shape
    assert two == 2
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
    b, c2, h, w = x.shape
    assert c2 % 2 == 0
    c = c2 // 2   
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    
def norm(x, m, v):
    return (x -m) * torch.rsqrt(v)

def unnorm(x, m, v):
    return m + x * torch.sqrt(v)
    
class Encoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=32):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=True)
    
    def forward(self, x, m, v):
        x = complex_to_chan_dim(x)
        m = m.view(1, -1, 1, 1)
        v = v.view(1, -1, 1, 1)
        x = self.conv(norm(x, m, v))
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels=32, out_channels=2):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=True)

    def forward(self, x, m, v):
        x = self.conv(x)
        m = m.view(1, -1, 1, 1)
        v = v.view(1, -1, 1, 1)
        x = unnorm(x, m, v)
        x = chan_complex_to_last_dim(x)
        return x

class Attention(nn.Module):
    def __init__(self, 
                 feature_chans : int = 32,
                 reducing_factor : int = 8):
        super().__init__()
        self.channels = feature_chans
        self.reducing_factor = reducing_factor
        self.conv1 = nn.Conv2d(feature_chans, feature_chans//reducing_factor, kernel_size = 1)
        self.conv2 = nn.Conv2d(feature_chans//reducing_factor, feature_chans, kernel_size = 1)
        
    def forward(self, f):
        t = F.relu(self.conv1(f))
        t = F.avg_pool2d(t, (1,1))
        t = F.relu(self.conv2(t))
        t = F.sigmoid(F.avg_pool2d(t, (1, 1)))
        return f * t 
    
class FNormUnet(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 32,
        out_chans: int = 32,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.reshape(b, 1, c * h * w)

        mean = x.mean(dim=2).view(b, 1, 1, 1)
        std = x.std(dim=2).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get shapes for unet and normalize
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        return x

def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )


    

class FVarNetBlock(nn.Module):

    def __init__(self, model: nn.Module, encoder : Encoder, decoder : Decoder, 
                feature_chans :int = 32):
        super().__init__()

        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.attention = Attention(feature_chans = feature_chans)
        self.dc_weight = nn.Parameter(torch.ones(1))


    def forward(
        self,
        current_fspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        mean : torch.Tensor,
        var : torch.Tensor,
    ) -> torch.Tensor:
        
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_fspace)
                
        
        soft_dc = self.dc_weight * self.encoder(
            sens_reduce(
                    torch.where(mask, 
                                sens_expand(self.decoder(current_fspace, mean, var), sens_maps) - ref_kspace, 
                                zero), 
                    sens_maps),
            mean, var)
        
        model_term = self.model(self.attention(current_fspace))        
        
        return current_fspace - soft_dc - model_term

class FVarNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 6,
        num_cascades_img : int = 6,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        
        feature_chans : int = 32
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()
        self.encoder = Encoder(out_channels = feature_chans)
        self.decoder = Decoder( in_channels = feature_chans)
        
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [FVarNetBlock(FNormUnet(chans, pools,  in_chans=feature_chans, out_chans=feature_chans), self.encoder, self.decoder, feature_chans = feature_chans) 
             for _ in range(num_cascades)]
        )
        self.cascades_img = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools, in_chans=2, out_chans=2)) for _ in range(num_cascades_img)]
        )
        self.current_epoch = 0
        
    def mean_var(self, image):
        image = complex_to_chan_dim(image)
        m = torch.mean(image, dim=[0,2,3])
        v = torch.var(image, dim=[0,2,3])
        image = chan_complex_to_last_dim(image)
        return m, v
    
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()
        image = sens_reduce(kspace_pred, sens_maps)
        
        mean, var = self.mean_var(image)
        
        fspace_pred = self.encoder(image, mean, var)
        
        for cascade in self.cascades:
            fspace_pred = cascade(fspace_pred, masked_kspace, mask, sens_maps, mean, var)
            
        kspace_pred = sens_expand(self.decoder(fspace_pred, mean, var), sens_maps)
                
        for cascade in self.cascades_img:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
            
        result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        
        height = result.shape[-2]
        width = result.shape[-1]
        
        return result[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]