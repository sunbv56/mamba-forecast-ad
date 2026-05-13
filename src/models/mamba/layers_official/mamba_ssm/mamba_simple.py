import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_casual_conv=True,
            use_fast_path=False,  # Fused kernel options
            layer_idx=None,
            dropout=0.,
            n_vars=0,
            VPT_mode=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.use_casual_conv = use_casual_conv
        self.layer_idx = layer_idx
        self.n_vars = n_vars
        self.dropout = dropout
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        if self.use_casual_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.activation = "silu"
        self.act = nn.SiLU()

        if self.dropout > 0 and not self.use_fast_path:
            self.x_proj = nn.Sequential(
                nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
                nn.Dropout(dropout),
            )
        else:
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, ids_restore=None):
        batch, seqlen, dim = hidden_states.shape
        
        # Mamba Inner logic simplified for portability
        # In a real environment with CUDA kernels, mamba_inner_fn is used.
        # Here we provide a fallback for CPU/Standard Torch.
        
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        
        if self.use_casual_conv:
            x = x.transpose(1, 2)
            x = self.act(self.conv1d(x)[..., :seqlen])
            x = x.transpose(1, 2)
            
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        
        A = -torch.exp(self.A_log.float())
        
        # Simple scan fallback if selective_scan_fn is not available or if running on CPU
        if selective_scan_fn is None or not x.is_cuda:
            # Simplified scan
            dt = F.softplus(dt)
            dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
            dB = torch.einsum("bld,bln->bldn", dt, B)
            
            # This is slow in pure torch, but works for CPU fallback
            h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
            outputs = []
            for t in range(seqlen):
                h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
                y = torch.einsum("bdn,bn->bd", h, C[:, t])
                outputs.append(y)
            y = torch.stack(outputs, dim=1)
        else:
            # Use fast CUDA kernels if available
            x = x.transpose(1, 2)
            dt = dt.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            y = selective_scan_fn(
                x, dt, A, B, C, self.D.float(), z=z.transpose(1, 2),
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True
            )
            y = y.transpose(1, 2)

        return self.out_proj(y * self.act(z) if selective_scan_fn is None else y)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, ids_restore=None
    ):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
            
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, ids_restore=ids_restore)
        return hidden_states, residual
