import torch 
import torch.nn as nn 
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_utils import ATTN_TYPE, WindowAttention2D, LayerNorm

from typing import Sequence, Optional

import math


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(2).unsqueeze(3)) + shift.unsqueeze(2).unsqueeze(3)


class ConvSwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Sequential(
            nn.Conv2d(dim, 2 * hidden_dim, kernel_size=1, bias=bias),
            nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1, groups=2 * hidden_dim)
        )
        self.w3 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class RealSTL(nn.Module):
    def __init__(
        self, dim: int, window_size: int, num_head: int, 
        attn_type: int, shift: bool,
        rank: Optional[int] = None, rib_hidden_dim: Optional[int] = None, rib_n_freqs: Optional[int] = None,
        exp_ratio: float | int = 2,
        attn_func: Optional[any] = None,
        gate_type: Optional[str] = None,
        drop_p = 0.0,
    ):
        super().__init__()
        
        self.norm_attn = LayerNorm(dim, use_affine=False)
        self.attn = WindowAttention2D(
            dim=dim, window_size=window_size, num_heads=num_head,
            attn_type=attn_type, shift=shift, rank=rank, 
            rib_hidden_dim=rib_hidden_dim, rib_n_freqs=rib_n_freqs,
            attn_func=attn_func, gate_type=gate_type
        )
        
        self.norm_ffn = LayerNorm(dim, use_affine=False)
        self.ffn = ConvSwiGLUFFN(dim, int(dim * exp_ratio), drop=drop_p)
            
    def forward(
            self, x: torch.Tensor, 
            attn_shift, attn_scale, attn_gate, ffn_shift, ffn_scale, ffn_gate
        ) -> torch.Tensor:
        x = x + self.attn(modulate(self.norm_attn(x), attn_shift, attn_scale)) * attn_gate.unsqueeze(2).unsqueeze(3)
        x = x + self.ffn(modulate(self.norm_ffn(x), ffn_shift, ffn_scale)) * ffn_gate.unsqueeze(2).unsqueeze(3)
        return x


class NNx4Upsampler(nn.Sequential):
    def __init__(self, dim_in, dim_up):
        super().__init__(
            nn.Conv2d(dim_in, dim_up, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dim_up, dim_up, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dim_up, dim_up, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim_up, 3, kernel_size=3, stride=1, padding=1),
        )


class RealSSTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        window_sizes: Sequence[int],
        num_heads: Sequence[int],
        attn_type: str,
        ranks: Sequence[Optional[int]],
        rib_hidden_dim: Optional[int] = None,
        rib_n_freqs: Optional[int] = None,
        exp_ratio: int | float = 2,
        attn_func: Optional[any] = None,
        gate_type: Optional[str] = None,
        fuse_lr: bool = False,
        **kwargs
    ):
        super().__init__()
        if not (len(window_sizes) == len(num_heads) == len(ranks)):
            raise ValueError(
                'window_sizes, num_heads, and ranks must have identical lengths: '
                f'{len(window_sizes)}, {len(num_heads)}, {len(ranks)}'
            )

        if fuse_lr:
            self.fuse_lr = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0)

        use_shift = all(window_size == window_sizes[0] for window_size in window_sizes)
        self.blocks = nn.ModuleList([
            RealSTL(
                dim=in_channels,
                window_size=window_size,
                num_head=num_head,
                attn_type=attn_type,
                shift=(block_idx % 2 == 1) and use_shift,
                rank=rank,
                rib_hidden_dim=rib_hidden_dim,
                rib_n_freqs=rib_n_freqs,
                exp_ratio=exp_ratio,
                attn_func=attn_func,
                gate_type=gate_type,
                drop_p=kwargs.get('drop_p', 0.0),
            )
            for block_idx, (window_size, num_head, rank) in enumerate(zip(window_sizes, num_heads, ranks))
        ])
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.adaLN_modulation = nn.Linear(emb_channels, 6 * in_channels, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)
        self.gamma = nn.Parameter(torch.full((1, in_channels, 1, 1), 1e-3))

    def forward(self, x: torch.Tensor, lr = None, embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x

        attn_shift, attn_scale, attn_gate, ffn_shift, ffn_scale, ffn_gate = self.adaLN_modulation(embed).chunk(6, dim=-1)

        if lr is not None:
            x = self.fuse_lr(
                torch.cat([x, lr], dim=1)
            )

        for block in self.blocks:
            x = block(x, attn_shift, attn_scale, attn_gate, ffn_shift, ffn_scale, ffn_gate)
        x = self.out_conv(x) * self.gamma
        return residual + x


class SSTReal(nn.Module):
    def __init__(
        self, **model_kwargs
    ):
        super().__init__()
        dim = model_kwargs['dim']
        intermediate_dim = model_kwargs['intermediate_dim']
        emb_channels = model_kwargs['emb_channels']
        noise_channels = model_kwargs['noise_channels']

        self.fuse_lr_till = model_kwargs['fuse_lr_till']
        self.ig_idx = model_kwargs['ig_idx']

        self.map_noise = TimestepEmbedder(hidden_size=emb_channels, frequency_embedding_size=noise_channels)

        self.proj_in = nn.Conv2d(3 * 4 * 4, dim, kernel_size=3, stride=1, padding=1)
        self.proj_lr = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1)
        self.proj_body = nn.ModuleList([
            RealSSTBlock(in_channels=dim, fuse_lr=i <= self.fuse_lr_till, **model_kwargs)
            for i in range(model_kwargs['num_blocks'])
        ])

        self.up_i = NNx4Upsampler(dim, intermediate_dim)
        self.up_f = NNx4Upsampler(dim, intermediate_dim)

    def no_weight_decay(self):
        no_decay = set()

        for module_name, module in self.named_modules():
            if isinstance(module, NNx4Upsampler):
                for param_name, _ in module.named_parameters(prefix=module_name, recurse=True):
                    if param_name.endswith('.bias'):
                        no_decay.add(param_name)

                for child_name, child in reversed(list(module.named_children())):
                    if isinstance(child, nn.Conv2d) and child.out_channels == 3:
                        no_decay.add(f'{module_name}.{child_name}.weight')
                        if child.bias is not None:
                            no_decay.add(f'{module_name}.{child_name}.bias')
                        break

            if isinstance(module, WindowAttention2D):
                for param_name in ('to_hidden', 'hidden_b', 'to_q', 'to_k'):  # RIB-related parameters
                    if hasattr(module, param_name):
                        no_decay.add(f'{module_name}.{param_name}')

        for param_name, _ in self.named_parameters():
            if (
                param_name == 'proj_lr.bias'
                or param_name.endswith('.adaLN_modulation.bias')
                or param_name.endswith('.fuse_lr.bias')
            ):
                no_decay.add(param_name)

        return no_decay

    def forward(self, x, noise, lq, **kwargs):
        noise_emb = self.map_noise(noise)
        x = self.proj_in(F.pixel_unshuffle(x, downscale_factor=4))
        feat_skip = x
        lr_emb = self.proj_lr(lq)
        for idx, block in enumerate(self.proj_body):
            if idx <= self.fuse_lr_till:
                x = block(x, lr=lr_emb, embed=noise_emb)
            else:
                x = block(x, embed=noise_emb)
            if idx == self.ig_idx:
                x_i = self.up_i(x + feat_skip)
        x_f = self.up_f(x + feat_skip)
        return x_i, x_f


@ARCH_REGISTRY.register()
class EDMUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        up_list         = None,
        down_list       = None,
        use_skip        = True,
        sampling_ig_lambda = 1.,
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.up_list = up_list
        self.down_list = down_list
        self.use_skip = use_skip
        self.sampling_ig_lambda = sampling_ig_lambda
        self.model = SSTReal(
            **model_kwargs
        )
    
    def forward(self, x, sigma, lq, lq_up=None, force_fp32=False, **model_kwargs):
        return_pair = model_kwargs.get('return_pair', False)

        x = x.to(torch.float32)
        x_lr = lq
        sigma = sigma.reshape(-1, 1, 1, 1).to(torch.float32)
        model_dtype = torch.float32
        if self.use_skip:
            c_skip = self.sigma_data ** 2 / (((sigma / 0.1)) ** 2 + self.sigma_data ** 2)
            c_out = (sigma / 0.1) / ((sigma / 0.1) ** 2 + self.sigma_data ** 2).sqrt()
        else:
            c_skip = self.sigma_data ** 2 / (((sigma / 0.001)) ** 2 + self.sigma_data ** 2)
            c_out = (sigma / 0.001) / ((sigma / 0.001) ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = ((sigma + 0.002).log() / 4).to(model_dtype)
        c_skip = c_skip.to(torch.float32)
        c_out = c_out.to(torch.float32)

        model_input = x.to(model_dtype)
        
        x_i, x_f = self.model(model_input, c_noise.flatten(), lq=x_lr)

        D_x_i = c_skip * x + c_out * x_i.to(torch.float32)
        D_x_f = c_skip * x + c_out * x_f.to(torch.float32)

        if self.training or return_pair:
            return D_x_i, D_x_f

        return D_x_i + self.sampling_ig_lambda * (D_x_f - D_x_i)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

if __name__ == '__main__':
    # network_g:
    # type: EDMUNet
    # img_resolution: 192
    # img_channels: 3
    # scale: 4
    # sigma_data: 0.5
    # sigma_min: 0
    # use_skip: True
    # sampling_ig_lambda: 1.0

    # dim: 192
    # intermediate_dim: 96
    # emb_channels: 768
    # noise_channels: 192
    # num_blocks: 14
    # fuse_lr_till: 6  
    # ig_idx: 6       
    # attn_type: RIB
    # window_sizes: [16, 32, 64, 16, 32, 64]
    # num_heads: [6, 6, 6, 6, 6, 6]
    # ranks: [16, 16, 16, 32, 32, 32]
    # rib_hidden_dim: 32
    # rib_n_freqs: 10
    # exp_ratio: 3
    # gate_type: DWC
    # drop_p: 0.05
    model_kwargs = dict(
        dim=192,
        intermediate_dim=96,
        emb_channels=768,
        noise_channels=192,
        num_blocks=14,
        fuse_lr_till=6,
        ig_idx=6,
        attn_type='RIB',
        window_sizes=[16, 32, 64, 16, 32, 64],
        num_heads=[6, 6, 6, 6, 6, 6],
        ranks=[16, 16, 16, 32, 32, 32],
        rib_hidden_dim=32,
        rib_n_freqs=10,
        exp_ratio=3,
        gate_type='DWC',
        drop_p=0.05,
    )
    model = SSTReal(**model_kwargs)

    # x = torch.randn(2, 3 * 4 * 4, 64, 64)
    # x = torch.randn(1, 3 * 4 * 4, 128, 128)
    # noise = torch.tensor([0.5])
    # lq = torch.randn(1, 3, 128, 128)
    
    from scripts.test_direct_metrics import test_direct_metrics_edm
    test_direct_metrics_edm(model, (1, 3, 1024, 1024), (1, 3, 256, 256))