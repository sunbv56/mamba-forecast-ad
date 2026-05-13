import math
from functools import partial
import copy
import torch
import torch.nn as nn
from einops import rearrange, reduce

from .mamba_simple import Mamba, Block
from ...utils_official.masking import random_shuffle, unshuffle

try:
    from python_tsp.heuristics import solve_tsp_simulated_annealing
except ImportError:
    solve_tsp_simulated_annealing = None


def create_block(
        d_model,
        d_intermediate,
        ssm_cfg=None,
        n_vars=None,
        VPT_mode=0,
        dropout=0.,
        use_casual_conv=True,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
):
    ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
    ssm_layer = ssm_cfg.pop("layer", "Mamba1")
    factory_kwargs = {"device": device, "dtype": dtype}
    
    mixer_cls = partial(
        Mamba, # We currently only support Mamba1 in this port
        layer_idx=layer_idx,
        dropout=dropout, use_casual_conv=use_casual_conv, n_vars=n_vars, VPT_mode=VPT_mode,
        **ssm_cfg,
        **factory_kwargs
    )
    norm_cls = partial(nn.LayerNorm, eps=norm_epsilon, **factory_kwargs)
    
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class MixerTSModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            d_intermediate: int = 0,
            ssm_cfg=None,
            n_vars=None,
            VPT_mode=0,
            ATSP_solver='SA',
            dropout=0.,
            use_casual_conv: bool = True,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.n_vars = n_vars
        self.VPT_mode = VPT_mode
        self.ATSP_solver = ATSP_solver
        self.d_model = d_model
        self.ids_shuffle = None

        # VAST: Variable-Aware Scan along Time
        self.register_buffer("adjacency_matrix", torch.zeros(n_vars, n_vars))
        self.register_buffer("count_matrix", torch.zeros(n_vars, n_vars))
        self.register_buffer("ending_points", torch.zeros(n_vars))

        self.layers = nn.ModuleList([
            create_block(
                d_model,
                d_intermediate=d_intermediate,
                ssm_cfg=ssm_cfg,
                n_vars=self.n_vars,
                VPT_mode=self.VPT_mode,
                dropout=dropout,
                use_casual_conv=use_casual_conv,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                **factory_kwargs,
            ) for i in range(n_layer)
        ])

        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon, **factory_kwargs)

    def forward(self, x, inference_params=None, **mixer_kwargs):
        n_vars_cur = self.n_vars
        
        # Variable Permutation / Scanning Logic
        if self.VPT_mode == 1:
            if self.training:
                # Alternating VST: b (l c) d
                x = rearrange(x, 'b (l c) d -> b c (l d)', c=n_vars_cur)
                x, ids_shuffle, ids_restore = random_shuffle(x, mask_ratio=0, return_ids_shuffle=True)
                
                # Record transitions for VAST
                ids_shuffle_ = torch.cat([ids_shuffle[:, [0]], ids_shuffle], dim=-1)
                self.transition_tuple = torch.stack([ids_shuffle_[:, :-1], ids_shuffle_[:, 1:]], dim=-1)
                
                x = rearrange(x, 'b c (l d) -> b (l c) d', d=self.d_model)
                self.ids_shuffle = None
            else:
                # Alternating VST: b (l c) d
                x = rearrange(x, 'b (l c) d -> b c (l d)', c=n_vars_cur)
                if self.ids_shuffle is None:
                    # Solve ATSP for scanning order
                    if n_vars_cur > 2 and solve_tsp_simulated_annealing is not None:
                        adj = self.get_adjacency_matrix().cpu().numpy()
                        # Simplified: Use fixed or greedy if TSP solver not available
                        try:
                            path, _ = solve_tsp_simulated_annealing(adj)
                            ids_shuffle = torch.tensor(path, device=x.device)
                        except Exception:
                            ids_shuffle = torch.arange(n_vars_cur, device=x.device)
                    else:
                        ids_shuffle = torch.arange(n_vars_cur, device=x.device)
                    self.ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)
                
                x, ids_restore = random_shuffle(x, ids_shuffle=self.ids_shuffle)
                x = rearrange(x, 'b c (l d) -> b (l c) d', d=self.d_model)
        else:
            ids_restore = None

        hidden_states = x
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual)

        if self.VPT_mode == 1:
            hidden_states = rearrange(hidden_states, 'b (l c) d -> b c (l d)', c=n_vars_cur)
            hidden_states = unshuffle(hidden_states, ids_restore)
            hidden_states = rearrange(hidden_states, 'b c (l d) -> b (l c) d', d=self.d_model)

        return hidden_states, None

    def get_adjacency_matrix(self):
        adj = self.adjacency_matrix / (self.count_matrix + 1e-8)
        return adj + torch.abs(adj.min()) + 1e-7

    def batch_update_state(self, cost_tensor):
        cost_tensor = cost_tensor.detach()
        if cost_tensor.dim() > 1:
            # Average over all dimensions except Batch
            cost_tensor = cost_tensor.mean(dim=tuple(range(1, cost_tensor.dim())))
        
        B, C = cost_tensor.size(0), self.adjacency_matrix.size(0)
        indices = self.transition_tuple[:, :, 0] * C + self.transition_tuple[:, :, 1]
        self.adjacency_matrix.view(-1).scatter_add_(0, indices.view(-1), cost_tensor.repeat(indices.shape[-1]))
        self.count_matrix.view(-1).scatter_add_(0, indices.view(-1), torch.ones_like(cost_tensor).repeat(indices.shape[-1]))
