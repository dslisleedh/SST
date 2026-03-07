import torch 
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange 

from torch.nn.attention.flex_attention import flex_attention

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_utils import ATTN_TYPE, WindowAttention2D, LayerNorm, UPSAMPLER_TYPE, ImageArchitecture

from typing import Sequence, Optional


class ConvFFN(nn.Module):
    def __init__(self, dim: int, exp_ratio: float | int, kernel_size: int = 3):
        super().__init__()
        d_in = dim
        d_hidden = int(dim * exp_ratio)
        
        self.proj = nn.Conv2d(d_in, d_hidden, kernel_size=1, stride=1, padding=0)
        self.dwc = nn.Conv2d(d_hidden, d_hidden, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=d_hidden)
        self.agg = nn.Conv2d(d_hidden, d_in, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj(x))
        x = x + F.gelu(self.dwc(x))
        x = self.agg(x)
        return x


class STL(nn.Module):
    def __init__(
        self, dim: int, window_size: int, num_head: int, 
        attn_type: int, shift: bool,
        rank: Optional[int] = None, rib_hidden_dim: Optional[int] = None, rib_n_freqs: Optional[int] = None,
        exp_ratio: float | int = 2,
        attn_func: Optional[any] = None,
        gate_type: Optional[str] = None,
    ):
        super().__init__()
        if attn_type == 'CPE':
            attn_type = 'NoPE'
            self.use_cpe = True
            self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        else:
            self.use_cpe = False
        
        self.norm_attn = LayerNorm(dim)
        self.attn = WindowAttention2D(
            dim=dim, window_size=window_size, num_heads=num_head,
            attn_type=attn_type, shift=shift, rank=rank, 
            rib_hidden_dim=rib_hidden_dim, rib_n_freqs=rib_n_freqs,
            attn_func=attn_func, gate_type=gate_type
        )
        
        self.norm_ffn = LayerNorm(dim)
        self.ffn = ConvFFN(dim, exp_ratio)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cpe:
            x = x + self.cpe(x)
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x
    

class SSTB(nn.Sequential):
    def __init__(
        self, dim: int, window_sizes: Sequence[int], num_heads: Sequence[int],
        attn_type: str, ranks: Sequence[Optional[int]],
        rib_hidden_dim: Optional[int] = None, rib_n_freqs: Optional[int] = None,
        exp_ratio: float | int = 2,
        attn_func: Optional[any] = None,
        gate_type: Optional[str] = None,
    ):
        use_shift = all([ws == window_sizes[0] for ws in window_sizes])
        super().__init__(*[
            STL(
                dim=dim, window_size=ws, num_head=nh,
                attn_type=attn_type, shift=(i % 2 == 1) and use_shift,
                rank=rank, rib_hidden_dim=rib_hidden_dim, rib_n_freqs=rib_n_freqs,
                exp_ratio=exp_ratio, attn_func=attn_func,
                gate_type=gate_type, 
            ) for i, (ws, nh, rank) in enumerate(zip(window_sizes, num_heads, ranks)) 
        ] + [
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super().forward(x)


@ARCH_REGISTRY.register()
class SST(ImageArchitecture):
    def __init__(
        self,
        dim: int, 
        window_sizes: Sequence[int],
        num_heads: Sequence[int],
        n_blocks: int,
        exp_ratio: float | int = 2,
        attn_type: ATTN_TYPE = 'RIB',
        rib_hidden_dim: Optional[int] = 64,
        rib_n_freqs: Optional[int] = 10,
        upscaling_factor: int = 2,
        upsampler_type: UPSAMPLER_TYPE = 'pixelshuffle_direct',
        ranks: Sequence[Optional[int]] = None,
        gate_type: Optional[str] = None,
        intermediate_dim: int = 64,
    ):
        super().__init__()
        assert len(window_sizes) == len(num_heads), "window_sizes and num_heads must have the same length."
        if attn_type in ['RIB', 'RIBSiren']:
            assert len(window_sizes) == len(ranks), "window_sizes and ranks must have the same length."
        else:
            if ranks is None:
                ranks = [None] * len(window_sizes)
        
        attn_func = torch.compile(flex_attention, dynamic=True) if attn_type == 'Flex' else None
        self.proj = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1)
        self.body = nn.Sequential(*[
            SSTB(
                dim=dim, window_sizes=window_sizes, num_heads=num_heads,
                attn_type=attn_type, ranks=ranks,
                rib_hidden_dim=rib_hidden_dim, rib_n_freqs=rib_n_freqs,
                exp_ratio=exp_ratio, attn_func=attn_func,
                gate_type=gate_type
            ) for _ in range(n_blocks)
        ] + [
            LayerNorm(dim),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        ])
            
        self.build_upsampler(dim, upscaling_factor, upsampler_type, intermediate_dim=intermediate_dim)
        self.init_weight()
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)
        feat = feat + self.body(feat)
        x = self.upsampler(feat, x)
        return x
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.Conv2d):
                if m.weight.shape[-1] == 1:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    pass  # Use standard initialization for 3x3 convs
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


if __name__ == '__main__':
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
    
    # light
    model_kwargs = {
        'dim': 48,
        'window_sizes': [8, 16, 32, 16, 32, 64],
        # 'window_sizes': [16, 32, 48, 32, 48, 96],
        'num_heads': [3, 3, 3, 3, 3, 3],
        'ranks': [16, 16, 16, 24, 24, 24],
        'n_blocks': 5,
        'exp_ratio': 1.25,
        'attn_type': 'FlashBias',
        'rib_hidden_dim': 32,
        'rib_n_freqs': 10,
        'upscaling_factor': upsampling_factor,
        'upsampler_type': 'pixelshuffle_direct',
        'gate_type': 'DWC',
    }
    # Base
    model_kwargs = {
        'dim': 180,
        # 'window_sizes': [16, 32, 64, 16, 32, 64],
        'window_sizes': [16, 32, 48, 32, 48, 96],
        'num_heads': [6, 6, 6, 6, 6, 6],
        'ranks': [18, 18, 18, 34, 34, 34],
        'n_blocks': 6,
        'exp_ratio': 1.25,
        'attn_type': 'RIB',
        'rib_hidden_dim': 32,
        'rib_n_freqs': 10,
        'upscaling_factor': upsampling_factor,
        'upsampler_type': 'pixelshuffle',
        'gate_type': 'DWC',
    }
    # Large
    # model_kwargs = {
    #     'dim': 192,
    #     # 'window_sizes': [16, 32, 64, 16, 32, 64],
    #     'window_sizes': [16, 32, 48, 32, 48, 96],
    #     'num_heads': [6, 6, 6, 6, 6, 6],
    #     'ranks': [16, 16, 16, 32, 32, 32],
    #     'n_blocks': 8,
    #     'exp_ratio': 2,
    #     'attn_type': 'RoPEViT',
    #     'rib_hidden_dim': 32,
    #     'rib_n_freqs': 10,
    #     'upscaling_factor': upsampling_factor,
    #     'upsampler_type': 'pixelshuffle',
    #     'gate_type': 'DWC',
    #     'intermediate_dim': 96,
    # }
    model = SST(
        **model_kwargs
    )
    
    print(model)
    
    shape = (batch_size, 3, height // upsampling_factor, width // upsampling_factor)
    
    print('Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000, 'M')
    
    test_direct_metrics(model, shape, use_float16=False, n_repeat=100)
    
    # with torch.no_grad():
    #     x = torch.randn(shape)
    #     x = x.cuda()
    #     model = model.cuda()
    #     model = model.eval()
    #     flops = FlopCountAnalysis(model, x)
    #     print(f'FLOPs (G): {flops.total()/1e9:.2f} G')
    #     print(f'Params: {sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1000}K')
