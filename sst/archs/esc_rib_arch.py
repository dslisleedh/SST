import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange 

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_utils import ATTN_TYPE, WindowAttention2D, UPSAMPLER_TYPE, ImageArchitecture

from torch.nn.attention.flex_attention import flex_attention
from typing import Optional, Sequence


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            if self.training:
                return F.layer_norm(x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
            else:
                return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int, kernel_size: int = 13):
        super().__init__()
        self.pdim = pdim
        self.lk_size = kernel_size
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
            
            # Dynamic Conv
            bs = x1.shape[0]
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(-1, 1, self.sk_size, self.sk_size)
            x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')
            x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size//2, groups=bs * self.pdim)
            x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim)
            
            # Static LK Conv + Dynamic Conv
            x1 = F.conv2d(x1, lk_filter, stride=1, padding=self.lk_size // 2) + x1_
            
            x = torch.cat([x1, x2], dim=1)
        else:
            # for GPU
            dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(self.pdim, 1, self.sk_size, self.sk_size) 
            x[:, :self.pdim] = F.conv2d(x[:, :self.pdim], lk_filter, stride=1, padding=self.lk_size // 2) \
                + F.conv2d(x[:, :self.pdim], dynamic_kernel, stride=1, padding=self.sk_size // 2, groups=self.pdim)
            
            # For Mobile Conversion, uncomment the following code
            # x_1, x_2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)
            # dynamic_kernel = self.dwc_proj(x_1).reshape(16, 1, 3, 3)
            # x_1 = F.conv2d(x_1, lk_filter, stride=1, padding=13 // 2) + F.conv2d(x_1, dynamic_kernel, stride=1, padding=1, groups=16)
            # x = torch.cat([x_1, x_2], dim=1)
        return x
    
    def extra_repr(self):
        return f'pdim={self.pdim}'
    

class ConvAttnWrapper(nn.Module):
    def __init__(self, dim: int, pdim: int, kernel_size: int = 13):
        super().__init__()
        self.plk = ConvolutionalAttention(pdim, kernel_size)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        x = self.plk(x, lk_filter)
        x = self.aggr(x)
        return x 


class ConvFFN(nn.Module):
    def __init__(self, dim: int, kernel_size: int, exp_ratio: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, int(dim*exp_ratio), 1, 1, 0)
        self.dwc = nn.Conv2d(int(dim*exp_ratio), int(dim*exp_ratio), kernel_size, 1, kernel_size//2, groups=int(dim*exp_ratio))
        self.aggr = nn.Conv2d(int(dim*exp_ratio), dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj(x))
        x = F.gelu(self.dwc(x)) + x
        x = self.aggr(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim: int, pdim: int, conv_blocks: int, 
            kernel_size: int, window_size: int, num_heads: int, exp_ratio: int, 
            attn_func=None, attn_type: ATTN_TYPE = 'Flex', use_ln: bool = False,
            flashbias_rank: Optional[int] = None,
            rank: Optional[int] = None,
            rib_hidden_dim: Optional[int] = None,
            rib_n_freqs: Optional[int] = None,
            gate_type: Optional[str] = None,
        ):
        super().__init__()
        self.ln_proj = LayerNorm(dim)
        self.proj = ConvFFN(dim, 3, 4)

        self.ln_attn = LayerNorm(dim) 
        self.attn = WindowAttention2D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attn_type=attn_type,
            rank=flashbias_rank if rank is None else rank,
            attn_func=attn_func,
            shift=False,
            rib_hidden_dim=rib_hidden_dim,
            rib_n_freqs=rib_n_freqs,
            gate_type=gate_type,
        )
        
        self.lns = nn.ModuleList([LayerNorm(dim) if use_ln else nn.Identity() for _ in range(conv_blocks)])
        self.pconvs = nn.ModuleList([ConvAttnWrapper(dim, pdim, kernel_size) for _ in range(conv_blocks)])
        self.convffns = nn.ModuleList([ConvFFN(dim, 3, exp_ratio) for _ in range(conv_blocks)])
        
        self.ln_out = LayerNorm(dim)
        self.conv_out = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, plk_filter: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.ln_proj(x)
        x = self.proj(x)
        x = x + self.attn(self.ln_attn(x))
        for ln, pconv, convffn in zip(self.lns, self.pconvs, self.convffns):
            x = x + pconv(convffn(ln(x)), plk_filter)
        x = self.conv_out(self.ln_out(x))
        return x + skip


# To enhance LK's structural inductive bias, we use Feature-level Geometric Re-parameterization
#  as proposed in https://github.com/dslisleedh/IGConv
def _geo_ensemble(k):
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])
    k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8
    return k


@ARCH_REGISTRY.register()
class ESCRIB(ImageArchitecture):
    def __init__(
        self, dim: int, pdim: int, kernel_size: int,
        n_blocks: int, conv_blocks: int, window_size: int, num_heads: int,
        upscaling_factor: int, exp_ratio: int = 2, attn_type: ATTN_TYPE = 'Flex',
        use_ln: bool = False, flashbias_rank: Optional[int] = None,
        rank: Optional[int] = None,
        rib_hidden_dim: Optional[int] = 64,
        rib_n_freqs: Optional[int] = 10,
        gate_type: Optional[str] = None,
        upsampler_type: UPSAMPLER_TYPE = 'pixelshuffle_direct',
        intermediate_dim: int = 64,
    ):
        super().__init__()
        if attn_type == 'Flex':
            attn_func = torch.compile(flex_attention, dynamic=True)
            
        self.plk_func = _geo_ensemble
            
        self.plk_filter = nn.Parameter(torch.randn(pdim, pdim, kernel_size, kernel_size))
        # Initializing LK filters using orthogonal initialization is important for stabilizing early training phase.
        torch.nn.init.orthogonal_(self.plk_filter)  
        
        self.proj = nn.Conv2d(3, dim, 3, 1, 1)
        self.blocks = nn.ModuleList([
            Block(
                dim, pdim, conv_blocks, 
                kernel_size, window_size, num_heads, exp_ratio,
                attn_func, attn_type, use_ln=use_ln,
                flashbias_rank=flashbias_rank, rank=rank,
                rib_hidden_dim=rib_hidden_dim, rib_n_freqs=rib_n_freqs,
                gate_type=gate_type,
            ) for _ in range(n_blocks)
        ])
        # self.last = nn.Conv2d(dim, dim, 3, 1, 1)
        self.last = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.build_upsampler(dim, upscaling_factor, upsampler_type, intermediate_dim=intermediate_dim)
        
    @torch.no_grad()
    def convert(self):
        self.plk_filter = nn.Parameter(self.plk_func(self.plk_filter))
        self.plk_func = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)
        skip = feat
        plk_filter = self.plk_func(self.plk_filter)
        for block in self.blocks:
            feat = block(feat, plk_filter)
        feat = self.last(feat) + skip
        return self.upsampler(feat, x)


if __name__== '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    import numpy as np
    from scripts.test_direct_metrics import test_direct_metrics
    
    test_size = 'HD'
    # test_size = 'FHD'
    # test_size = '4K'

    height = 720 if test_size == 'HD' else 1080 if test_size == 'FHD' else 2160
    width = 1280 if test_size == 'HD' else 1920 if test_size == 'FHD' else 3840
    upsampling_factor = 4
    batch_size = 1
    
    # Base
    model_kwargs = {
        'dim': 64,
        'pdim': 16,
        'kernel_size': 13, 
        'n_blocks': 5,
        'conv_blocks': 5,
        'window_size': 64,
        'num_heads': 4,
        'upscaling_factor': upsampling_factor,
        'exp_ratio': 1.25,
        'attn_type': 'RIB',
        'rank': 16,
        'rib_hidden_dim': 32,
        'rib_n_freqs': 10,
        'gate_type': 'DWC',
        'upsampler_type': 'pixelshuffle_direct',
    }
    shape = (batch_size, 3, height // upsampling_factor, width // upsampling_factor)
    model = ESCRIB(**model_kwargs)
    print(model)
    
    model.convert()
        
    test_direct_metrics(model, shape, use_float16=False, n_repeat=100)

    with torch.inference_mode():
        x = torch.randn(shape)
        x = x.cuda()
        model = model.cuda()
        model = model.eval()
        flops = FlopCountAnalysis(model, x)
        print(f'FLOPs: {flops.total()/1e9:.2f} G')
        print(f'Params: {sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1000}K')
