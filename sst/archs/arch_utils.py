import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange 

from basicsr.utils.registry import ARCH_REGISTRY

from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from typing import Optional, Sequence, Literal, Tuple
from copy import deepcopy
import math

from functools import partial

import warnings
warnings.simplefilter("default")


ATTN_TYPE = Literal['NoPE', 'Naive', 'SDPA', 'Flex', 'FlashBias', 'RIB', 'RIBSiren', 'RoPEViT']
UPSAMPLER_TYPE = Literal['pixelshuffle_direct', 'pixelshuffle', 'nn+conv']

try:
    from flash_attn_interface import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    score = q @ k.transpose(-2, -1) / q.shape[-1]**0.5
    score = score + bias
    score = F.softmax(score, dim=-1)
    out = score @ v
    return out


def apply_rpe(table: torch.Tensor, window_size: int):
    def bias_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int):
        q_h = q_idx // window_size
        q_w = q_idx % window_size
        k_h = kv_idx // window_size
        k_w = kv_idx % window_size
        rel_h = k_h - q_h + window_size - 1
        rel_w = k_w - q_w + window_size - 1
        rel_idx = rel_h * (2 * window_size - 1) + rel_w
        return score + table[h, rel_idx]
    return bias_mod


def feat_to_qkv(x: torch.Tensor, window_size: Sequence[int], heads: int):
    return rearrange(
        x, 'b (qkv heads c) (h wh) (w ww) -> qkv (b h w) heads (wh ww) c',
        heads=heads, wh=window_size[0], ww=window_size[1], qkv=3
    )


def feat_to_qkv_fa(x: torch.Tensor, window_size: Sequence[int], heads: int):
    return rearrange(
        x, 'b (qkv heads c) (h wh) (w ww) -> qkv (b h w) (wh ww) heads c',
        heads=heads, wh=window_size[0], ww=window_size[1], qkv=3
    )
    

def out_to_feat(x, window_size: Sequence[int], h_div: int, w_div: int):
    return rearrange(
        x, '(b h w) heads (wh ww) c -> b (heads c) (h wh) (w ww)',
        h=h_div, w=w_div, wh=window_size[0], ww=window_size[1]
    )
    
    
def out_to_feat_fa(x, window_size: Sequence[int], h_div: int, w_div: int):
    return rearrange(
        x, '(b h w) (wh ww) heads c -> b (heads c) (h wh) (w ww)',
        h=h_div, w=w_div, wh=window_size[0], ww=window_size[1]
    )


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def init_random_2d_freqs(head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    if head_dim % 4 != 0:
        raise ValueError(f"RoPEViT requires head_dim % 4 == 0, but got head_dim={head_dim}")

    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    for _ in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)  # (H, head_dim/2)
    freqs_y = torch.stack(freqs_y, dim=0)  # (H, head_dim/2)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)  # (2, H, head_dim/2)
    return freqs


def compute_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor):
    """
    freqs: (2, H, head_dim/2)
    t_x, t_y: (N,)
    return: (H, N, head_dim/2) complex
    """
    device_type = freqs.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        freqs_x = (t_x.float().unsqueeze(-1) @ freqs[0].float().unsqueeze(-2))  # (H, N, D/2)
        freqs_y = (t_y.float().unsqueeze(-1) @ freqs[1].float().unsqueeze(-2))  # (H, N, D/2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)     # complex
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    else:
        raise ValueError(f"Unexpected shape: freqs_cis={freqs_cis.shape}, x={x.shape}")
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class WindowAttention2D(nn.Module):
    def __init__(
            self, dim: int, window_size: int, num_heads: int,
            attn_type: ATTN_TYPE = 'Flex', rank: Optional[int] = None,
            attn_func=None, shift: bool = False,
            rib_hidden_dim: Optional[int] = None, rib_n_freqs: Optional[int] = None,
            gate_type=None
        ):
        super().__init__()
        self.dim = dim
        window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.window_size = window_size
        self.num_heads = num_heads
        
        assert dim % num_heads == 0, f'Embedding dimension {dim} should be divisible by number of heads {num_heads}.'

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)

        self.attn_type = attn_type
        if attn_func is None:
            if attn_type == 'Flex':
                raise ValueError('For Flex Attention, compiled attn_func must be provided.')
            if FLASH_ATTN_AVAILABLE and attn_type in ('FlashBias', 'RIB', 'RIBSiren', 'RoPEViT'):
                self.attn_func = partial(flash_attn_func, softmax_scale=1., causal=False)
            elif not FLASH_ATTN_AVAILABLE and attn_type in ('FlashBias', 'RIB', 'RIBSiren', 'RoPEViT'):
                self.attn_func = partial(F.scaled_dot_product_attention, dropout_p=0.0, is_causal=False, scale=1.0)
            else:
                self.attn_func = partial(F.scaled_dot_product_attention, dropout_p=0.0, is_causal=False)
        else:
            self.attn_func = attn_func
                
        self.qkv_func = feat_to_qkv_fa if (FLASH_ATTN_AVAILABLE and attn_type in ('FlashBias', 'RIB', 'RIBSiren', 'RoPEViT', 'NoPE')) else feat_to_qkv
        self.out_func = out_to_feat_fa if (FLASH_ATTN_AVAILABLE and attn_type in ('FlashBias', 'RIB', 'RIBSiren', 'RoPEViT', 'NoPE')) else out_to_feat

        if attn_type not in ('FlashBias', 'RIB', 'RIBSiren', 'RoPEViT', 'NoPE'):
            self.relative_position_bias = nn.Parameter(
                torch.randn(num_heads, (2 * window_size[0] - 1) * (2 * window_size[1] - 1)).to(torch.float32) * 0.001
            )
            if self.attn_type == 'Flex':
                self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
            else:  # Naive or SDPA
                self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)
        
        head_dim = dim // num_heads
        self.rank = 256 - head_dim if rank is None else rank

        if self.attn_type == 'FlashBias':
            self.flashbias_q = nn.Parameter(
                torch.zeros(num_heads, window_size[0] * window_size[1], self.rank)
            )
            self.flashbias_k = nn.Parameter(
                torch.zeros(num_heads, window_size[0] * window_size[1], self.rank)
            )
        
        if self.attn_type in ('RIB', 'RIBSiren'):
            Wh, Ww = window_size
            yy, xx = torch.meshgrid(torch.arange(Wh), torch.arange(Ww), indexing='ij')
            coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()  # (N,2)
            if Ww > 0:
                coords[:, 0] = (2.0 * (coords[:, 0] + 0.5) / Ww) - 1.0
            else:
                coords[:, 0] = 0
            if Wh > 0:
                coords[:, 1] = (2.0 * (coords[:, 1] + 0.5) / Wh) - 1.0
            else:
                coords[:, 1] = 0

            self.n_freqs = 0 if self.attn_type == 'RIBSiren' else 10 if rib_n_freqs is None else rib_n_freqs
            if self.n_freqs > 0:
                base_coords = coords.clone()
                for i in range(self.n_freqs):
                    freq = 2 ** i
                    coords = torch.cat([coords, torch.sin(base_coords * freq), torch.cos(base_coords * freq)], dim=-1)
            self.register_buffer("rib_coords", coords, persistent=False)
            
            n_input = 2 + 4 * self.n_freqs
            hidden_d = 32 if rib_hidden_dim is None else rib_hidden_dim

            self.to_hidden = nn.Parameter(torch.empty(n_input, hidden_d))
            self.hidden_b = nn.Parameter(torch.zeros(1, hidden_d))
            self.to_q = nn.Parameter(torch.empty(num_heads, hidden_d, self.rank))
            self.to_k = nn.Parameter(torch.empty(num_heads, hidden_d, self.rank))

            if self.attn_type == 'RIBSiren':
                self.rib_omega0 = 30.0 
                
            self._reset_rib_parameters(siren=(self.attn_type == 'RIBSiren'), omega0=getattr(self, "rib_omega0", 1.0))
        
        if self.attn_type == 'RoPEViT':
            if head_dim % 4 != 0:
                head_dim += 4 - (head_dim % 4)  

            Wh, Ww = window_size
            t_x, t_y = init_t_xy(end_x=Ww, end_y=Wh)
            self.register_buffer("rope_t_x", t_x, persistent=False)
            self.register_buffer("rope_t_y", t_y, persistent=False)

            rope_mixed = True  # Learnable frequencies
            rope_use_rpb = False  # Our goal is not to use RPB for Flash Attention
            rope_theta = 10.0  # recommended value
            self.rope_mixed = rope_mixed
            freqs = init_random_2d_freqs(
                head_dim=head_dim,
                num_heads=self.num_heads,
                theta=rope_theta,
                rotate=self.rope_mixed,
            )
            if self.rope_mixed:
                self.rope_freqs = nn.Parameter(freqs, requires_grad=True)
            else:
                self.register_buffer("rope_freqs", freqs, persistent=False)
                freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
                self.register_buffer("rope_freqs_cis", freqs_cis, persistent=False)
            self.rope_use_rpb = rope_use_rpb
        
        self.shift = shift

        self.gate_type = gate_type
        if gate_type is not None:
            if gate_type == 'Linear':
                self.gate = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Sigmoid()
                )
            elif gate_type == 'DWC':
                self.gate = nn.Sequential(
                    nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Sigmoid()
                )
            else:
                raise ValueError(f'Unsupported gate type: {gate_type}')

    def _reset_rib_parameters(self, siren: bool, omega0: float = 30.0):
        n_in = self.to_hidden.shape[0]

        if siren:
            bound = 1.0 / n_in
            nn.init.uniform_(self.to_hidden, -bound, bound)

            n_in2 = self.to_q.shape[1]  # hidden_d
            bound2 = math.sqrt(6.0 / n_in2) / omega0
            nn.init.uniform_(self.to_q, -bound2, bound2)
            nn.init.uniform_(self.to_k, -bound2, bound2)
        else:
            nn.init.normal_(self.to_hidden, mean=0.0, std=0.05)
            nn.init.normal_(self.to_q, mean=0.0, std=0.05)
            nn.init.normal_(self.to_k, mean=0.0, std=0.05)
            
    def _flash_cat_attn(self, q, k, v, pos_q, pos_k):
        if FLASH_ATTN_AVAILABLE:
            Bwin, Nq, H, D = q.shape
            Nkv = k.shape[1]
            pos_q = pos_q.transpose(1, 2)  # Bwin, N, heads, R
            pos_k = pos_k.transpose(1, 2)
        else:
            Bwin, H, Nq, D = q.shape
            Nkv = k.shape[2]
        R = pos_q.shape[-1]

        q = q * (D ** -0.5)
        pos_q = pos_q * (R ** -0.5)
        q_cat = torch.cat([q.to(torch.bfloat16), pos_q.to(torch.bfloat16)], dim=-1)
        k_cat = torch.cat([k.to(torch.bfloat16), pos_k.to(torch.bfloat16)], dim=-1)
        if FLASH_ATTN_AVAILABLE:
            v_cat = torch.cat([v.to(torch.bfloat16), torch.zeros((Bwin, Nkv, H, R), device=v.device, dtype=torch.bfloat16)], dim=-1)
        else:
            v_cat = torch.cat([v.to(torch.bfloat16), torch.zeros((Bwin, H, Nkv, R), device=v.device, dtype=torch.bfloat16)], dim=-1)
        
        d_total = q_cat.shape[-1]
        pad = (8 - (d_total % 8)) % 8
        if pad:
            q_cat = F.pad(q_cat, (0, pad), mode='constant', value=0)
            k_cat = F.pad(k_cat, (0, pad), mode='constant', value=0)
            v_cat = F.pad(v_cat, (0, pad), mode='constant', value=0)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            out = self.attn_func(
                q_cat.contiguous(),
                k_cat.contiguous(),
                v_cat.contiguous(),
            )[:, :, :, :D]
            # out = (
            #     F.softmax(q_cat @ k_cat.transpose(-2, -1), dim=-1) @ v_cat
            # )[:, :, :, :D]
        return out

    def _ropevit_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if FLASH_ATTN_AVAILABLE:
            q = q.transpose(1, 2)  # Bwin, heads, N, head_dim
            k = k.transpose(1, 2)

        headdim = q.shape[-1]
        if headdim % 4 != 0:
            pad = 4 - (headdim % 4)
            q = F.pad(q, (0, pad), mode='constant', value=0)
            k = F.pad(k, (0, pad), mode='constant', value=0)
            
        if self.rope_mixed:
            freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
        else:
            freqs_cis = self.rope_freqs_cis.to(device=q.device)

        q, k = apply_rotary_emb(q, k, freqs_cis)
        if headdim % 4 != 0:
            q = q[..., :headdim]
            k = k[..., :headdim]
        
        if FLASH_ATTN_AVAILABLE:
            q = q.transpose(1, 2)  # Bwin, N, heads, head_dim
            k = k.transpose(1, 2)
        
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            out = self.attn_func(
                q.to(torch.bfloat16).contiguous(),
                k.to(torch.bfloat16).contiguous(),
                v.to(torch.bfloat16).contiguous()
            )
        return out
    
    @staticmethod
    def create_table_idxs(window_size: int, heads: int):
        idxs_window = []
        for head in range(heads):
            for h in range(window_size ** 2):
                for w in range(window_size ** 2):
                    q_h = h // window_size
                    q_w = h % window_size
                    k_h = w // window_size
                    k_w = w % window_size
                    rel_h = k_h - q_h + window_size - 1
                    rel_w = k_w - q_w + window_size - 1
                    rel_idx = rel_h * (2 * window_size - 1) + rel_w
                    idxs_window.append((head, rel_idx))
        return torch.tensor(idxs_window, dtype=torch.long, requires_grad=False)

    def pad_to_win(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    def roll(self, x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
        if not self.shift:
            return x
        shift_size = (window_size[0] // 2, window_size[1] // 2)
        return torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))

    def unroll(self, x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
        if not self.shift:
            return x
        shift_size = (window_size[0] // 2, window_size[1] // 2)
        return torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        if self.gate_type is not None:
            gate = self.gate(x)
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)
        x = self.roll(x, self.window_size)
        h_div, w_div = x.shape[2] // self.window_size[0], x.shape[3] // self.window_size[1]

        qkv = self.to_qkv(x)
        dtype = qkv.dtype
        qkv = self.qkv_func(qkv, self.window_size, self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B*nwin, heads, N, head_dim) or (B*nwin, N, heads, head_dim)

        if self.attn_type == 'Flex':
            head_dim = q.shape[-1]
            target_dim = 1 << (head_dim - 1).bit_length()
            if head_dim != target_dim:
                if target_dim < head_dim:
                    target_dim = target_dim << 1
                q = F.pad(q, (0, target_dim - head_dim), mode='constant', value=0)
                k = F.pad(k, (0, target_dim - head_dim), mode='constant', value=0)
                v = F.pad(v, (0, target_dim - head_dim), mode='constant', value=0)
            q = q * (head_dim ** -0.5)
            out = self.attn_func(q, k, v, score_mod=self.get_rpe, scale=1.0)[:, :, :, :head_dim]

        elif self.attn_type == 'SDPA':
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1, self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1]
            )
            out = self.attn_func(q, k, v, attn_mask=bias)

        elif self.attn_type == 'Naive':
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1, self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1]
            )
            out = attention(q, k, v, bias)

        elif self.attn_type == 'FlashBias':
            Bwin = q.shape[0]

            q_bias = self.flashbias_q.to(dtype=q.dtype, device=q.device).unsqueeze(0).expand(Bwin, -1, -1, -1)
            k_bias = self.flashbias_k.to(dtype=k.dtype, device=k.device).unsqueeze(0).expand(Bwin, -1, -1, -1)
            
            out = self._flash_cat_attn(q, k, v, q_bias, k_bias)

        elif self.attn_type == 'RIB':
            Bwin = q.shape[0]

            intermediate = F.relu(
                self.rib_coords.to(dtype=torch.float32) @ self.to_hidden + self.hidden_b # N, hidden_d
            )
            q_pos = torch.einsum(
                'nd,hdr->hnr', intermediate, self.to_q.to(dtype=torch.float32)
            ).unsqueeze(0).expand(Bwin, -1, -1, -1)  # Bwin, H, Nq, R
            k_pos = torch.einsum(
                'nd,hdr->hnr', intermediate, self.to_k.to(dtype=torch.float32)
            ).unsqueeze(0).expand(Bwin, -1, -1, -1)  # Bwin, H, Nkv, R
            out = self._flash_cat_attn(q, k, v, q_pos, k_pos)

        elif self.attn_type == 'RIBSiren':
            Bwin = q.shape[0]

            coords = self.rib_coords.to(device=q.device, dtype=torch.float32)
            hidden_w = self.to_hidden.to(device=q.device, dtype=torch.float32)

            pre = coords @ hidden_w + self.hidden_b.to(device=q.device, dtype=torch.float32)  
            intermediate = torch.sin(self.rib_omega0 * pre)  
            to_q = self.to_q.to(device=q.device, dtype=torch.float32)
            to_k = self.to_k.to(device=q.device, dtype=torch.float32)

            q_pos = torch.einsum('nd,hdr->hnr', intermediate, to_q).to(dtype=q.dtype).unsqueeze(0).expand(Bwin, -1, -1, -1)
            k_pos = torch.einsum('nd,hdr->hnr', intermediate, to_k).to(dtype=k.dtype).unsqueeze(0).expand(Bwin, -1, -1, -1)

            out = self._flash_cat_attn(q, k, v, q_pos, k_pos)
            
        elif self.attn_type == 'RoPEViT':
            out = self._ropevit_attn(q, k, v)
        
        elif self.attn_type == 'NoPE':
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
                out = self.attn_func(
                    q.to(torch.bfloat16).contiguous(),
                    k.to(torch.bfloat16).contiguous(),
                    v.to(torch.bfloat16).contiguous()
                ).to(dtype)

        else:
            raise NotImplementedError(f'Attention type {self.attn_type} is not supported.')

        out = self.out_func(out, self.window_size, h_div, w_div)
        out = self.unroll(out, self.window_size).to(dtype)[:, :, :h, :w]
        if self.gate_type is not None:
            out = out * gate
        out = self.to_out(out)
        return out

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, attn_type={self.attn_type}, shift={self.shift}, USE_FLASH_ATTN_SOURCE_BUILD={FLASH_ATTN_AVAILABLE}'
       

class LayerNorm(nn.Module):
    """
    To compatible with 2D and 3D cases.
    """
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
                return F.layer_norm(x.transpose(1, -1).contiguous(), self.normalized_shape, self.weight, self.bias, self.eps).transpose(1, -1).contiguous()
            else:
                return F.layer_norm(x.transpose(1, -1), self.normalized_shape, self.weight, self.bias, self.eps).transpose(1, -1)


class Upsampler(nn.Module):
    """
    X4/PixelShuffle is different from the common implementations for weight interpolation initialization:
        - Common: Convx2 -> PSx2 -> Act -> Convx2 -> PSx2 -> Act -> Conv
        - Ours: Convx4 -> PSx4 -> Act -> Conv
    """
    def __init__(self, dim, upscaling_factor, upsampler_type: UPSAMPLER_TYPE, intermediate_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.upscaling_factor = upscaling_factor
        self.upsampler_type = upsampler_type
        
        self.target_weight_name = 'up_conv.weight'
        self.target_bias_name = 'up_conv.bias'
        
        if upsampler_type == 'pixelshuffle_direct':
            self.up_conv = nn.Conv2d(dim, 3 * (upscaling_factor ** 2), kernel_size=3, padding=1)
        
        elif upsampler_type == 'pixelshuffle':
            num_feat = intermediate_dim
            self.feature_conv = nn.Conv2d(dim, num_feat, kernel_size=3, padding=1)
            self.up_conv = nn.Conv2d(num_feat, num_feat * (upscaling_factor ** 2), kernel_size=3, padding=1)
            self.final_conv = nn.Conv2d(num_feat, 3, kernel_size=3, padding=1)
        
        elif upsampler_type == 'nn+conv':
            num_feat = intermediate_dim
            f = []
            f.extend(
                [
                    nn.Conv2d(dim, num_feat, kernel_size=3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1,inplace=True),
                ]
            )
            if (upscaling_factor & (upscaling_factor - 1)) == 0:
                for _ in range(int(math.log2(upscaling_factor))):
                    f.extend(
                        [
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
                            nn.LeakyReLU(negative_slope=0.1,inplace=True),
                        ]
                    )
            elif upscaling_factor == 3:
                f.extend(
                    [
                        nn.Upsample(scale_factor=3, mode='nearest'),
                        nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
                        nn.LeakyReLU(negative_slope=0.1,inplace=True),
                    ]
                )
            else:
                raise ValueError(
                    f"upscaling_factor {upscaling_factor} is not supported. Supported factors: 2^n and 3."
                )
            f.append(nn.Conv2d(num_feat, 3, kernel_size=3, padding=1))
            self.f = nn.Sequential(*f)
            self.f_img = nn.Sequential(
                nn.Conv2d(3, dim, 1),
                nn.Conv2d(dim, dim, kernel_size=7, padding=3),
                nn.LeakyReLU(negative_slope=0.1,inplace=True),
                nn.Conv2d(dim, dim, 1)
            )
        else:
            raise ValueError(f'upsampler_type {upsampler_type} is not supported.')
        
    def extra_repr(self) -> str:
        return f'upscaling_factor={self.upscaling_factor}, upsampler_type={self.upsampler_type}'
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.upsampler_type == 'pixelshuffle_direct':
            x = self.up_conv(x)
            if skip is not None:
                x = x + torch.repeat_interleave(
                    skip, repeats=self.upscaling_factor ** 2, dim=1
                )
            x = F.pixel_shuffle(x, self.upscaling_factor)
        elif self.upsampler_type == 'pixelshuffle':
            x = F.leaky_relu(self.feature_conv(x), negative_slope=0.1, inplace=True)
            x = self.up_conv(x)
            x = F.pixel_shuffle(x, self.upscaling_factor)
            x = self.final_conv(x)
            if skip is not None:
                x = x + F.interpolate(skip, scale_factor=self.upscaling_factor, mode='nearest')
        elif self.upsampler_type == 'nn+conv':
            x = self.f(x + self.f_img(skip)) if skip is not None else self.f(x)
        else:
            raise ValueError(f'upsampler_type {self.upsampler_type} is not supported.')
        return x
       

class ImageArchitecture(nn.Module):
    """
    For ease of upsampler management
    """
    def build_upsampler(self, dim, upscaling_factor, upsampler_type: UPSAMPLER_TYPE, intermediate_dim: int = 64):
        self.upsampler = Upsampler(dim, upscaling_factor, upsampler_type, intermediate_dim=intermediate_dim)
        
    def interpolate_upsampler(self, state_dict):
        if self.upsampler.upsampler_type != 'nn+conv':
            sd = deepcopy(state_dict)
            
            target_weight_name = f'upsampler.{self.upsampler.target_weight_name}'
            target_bias_name = f'upsampler.{self.upsampler.target_bias_name}'
            target_weight = sd[target_weight_name]
            target_bias = sd[target_bias_name]
            
            oc = target_weight.shape[0]
            ic = target_weight.shape[1]

            if self.upsampler.upsampler_type == 'pixelshuffle_direct':
                r2 = oc / 3
            elif self.upsampler.upsampler_type == 'pixelshuffle':
                r2 = oc / ic 
            else:
                raise ValueError

            sd_scale = int(round(math.sqrt(r2)))
            cur_scale = self.upsampler.upscaling_factor
            
            if sd_scale != cur_scale:
                from basicsr.utils import get_root_logger
                logger = get_root_logger()
                logger.info(
                    f'Interpolating Upsampler from x{sd_scale} to x{cur_scale}...'
                )
                
                out_dim = target_bias.shape[0] // (sd_scale ** 2)  # 3 or feat_dim
                def interpolate_kernel(kernel, scale_in, scale_out, out_dim):
                    _, _, kh, kw = kernel.shape
                    kernel = rearrange(kernel, '(dim rh rw) cin kh kw -> (cin kh kw) dim rh rw', dim=out_dim, rh=scale_in, rw=scale_in)
                    kernel = F.interpolate(kernel, size=(scale_out, scale_out), mode='bilinear', align_corners=False)
                    kernel = rearrange(kernel, '(cin kh kw) dim rh rw -> (dim rh rw) cin kh kw', kh=kh, kw=kw)
                    return kernel

                def interpolate_bias(bias, scale_in, scale_out, out_dim):
                    bias = rearrange(bias, '(dim rh rw) -> 1 dim rh rw', dim=out_dim, rh=scale_in, rw=scale_in)
                    bias = F.interpolate(bias, size=(scale_out, scale_out), mode='bilinear', align_corners=False)
                    bias = rearrange(bias, '1 dim rh rw -> (dim rh rw)')
                    return bias
                
                sd[target_weight_name] = interpolate_kernel(
                    target_weight, sd_scale, cur_scale, out_dim
                )
                sd[target_bias_name] = interpolate_bias(
                    target_bias, sd_scale, cur_scale, out_dim
                )
                
                return sd
            
        return state_dict
        
    def load_state_dict(self, state_dict, strict=True):
        state_dict = self.interpolate_upsampler(state_dict)
        super().load_state_dict(state_dict, strict)
        
