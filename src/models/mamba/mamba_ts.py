import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

from .layers import RevIN

class TemporalMambaBlock(nn.Module):
    """
    Temporal Mamba Block (TMB) as described in MambaTS.
    Simplified Mamba by removing/reducing spatial convolutions (d_conv=1)
    to focus on long-term temporal dependencies in time-series patches.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required for TemporalMambaBlock")
        
        # d_conv=1 removes the 1D convolution layer (makes it identity)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=1, 
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, D)
        res = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + res

class VariableAwareScanner:
    """
    Implements Variable-Aware Scanning (VAS) using ATSP.
    Calculates variable correlations and finds an optimal scanning order.
    """
    @staticmethod
    def solve_atsp(corr_matrix: np.ndarray) -> List[int]:
        """
        Solves ATSP using a greedy Nearest Neighbor heuristic.
        corr_matrix: (M, M) - higher values mean stronger correlation/similarity.
        We want to find an order that maximizes total correlation between adjacent variables.
        """
        num_vars = corr_matrix.shape[0]
        if num_vars <= 1:
            return [0]
            
        unvisited = list(range(num_vars))
        # Start with the variable that has the highest average correlation to others
        start_node = np.argmax(corr_matrix.sum(axis=1))
        order = [start_node]
        unvisited.remove(start_node)
        
        current = start_node
        while unvisited:
            # Find next node with highest correlation from current
            next_node = unvisited[np.argmax(corr_matrix[current, unvisited])]
            order.append(next_node)
            unvisited.remove(next_node)
            current = next_node
            
        return order

    @staticmethod
    def get_correlation_matrix(x: torch.Tensor) -> np.ndarray:
        """
        x: (Batch, M, L) -> returns correlation matrix (M, M)
        """
        B, M, L = x.shape
        if M <= 1:
            return np.eye(M)
            
        # Flatten batch and time to calculate overall correlation across variables
        # (M, B*L)
        x_flat = x.permute(1, 0, 2).reshape(M, -1)
        
        # Standardize
        x_std = (x_flat - x_flat.mean(dim=1, keepdim=True)) / (x_flat.std(dim=1, keepdim=True) + 1e-8)
        
        # Correlation matrix = (1/N) * X * X^T
        corr = torch.matmul(x_std, x_std.t()) / (x_flat.shape[1])
        return corr.detach().cpu().numpy()

class MambaTS(nn.Module):
    """
    MambaTS: Multi-variate Time Series Forecasting with Selective SSM.
    Ref: arXiv:2405.16440
    """
    def __init__(self, 
                 in_channels: int, 
                 lookback: int, 
                 forecast_len: int, 
                 patch_size: int = 64, 
                 stride: int = 32, 
                 d_model: int = 128, 
                 d_state: int = 16, 
                 n_layers: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1,
                 use_vas: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.lookback = lookback
        self.forecast_len = forecast_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.use_vas = use_vas
        
        # 1. RevIN
        self.revin = RevIN(num_features=in_channels, affine=True)
        
        # 2. Patching & Embedding
        self.patch_embedding = nn.Linear(patch_size, d_model)
        
        # 3. Temporal Mamba Blocks (TMB)
        self.tmb_layers = nn.ModuleList([
            TemporalMambaBlock(d_model=d_model, d_state=d_state, expand=expand)
            for _ in range(n_layers)
        ])
        
        # 4. Output Head
        # After scanning M variables * N patches, we have (B, M*N, D)
        # We need to output (B, M, forecast_len)
        num_patches = (lookback - patch_size) // stride + 1
        self.num_patches = num_patches
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model * num_patches, forecast_len)
        
        # Buffer for scan order (optional, could be instance-dependent)
        self.register_buffer("fixed_scan_order", torch.arange(in_channels))

    def forward(self, x: torch.Tensor, stats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, M, L) - M variables, L lookback
        returns: (B, M, forecast_len)
        """
        B, M, L = x.shape
        
        # 1. RevIN Norm
        x = self.revin(x, mode='norm')
        
        # 2. Variable Scanning Order
        if self.training:
            if self.use_vas:
                # Variable Permutation Training: random order to make model robust
                # but during evaluation/inference we use a stable order.
                scan_order = torch.randperm(M, device=x.device)
            else:
                scan_order = self.fixed_scan_order
        else:
            # Inference: use Fixed Order or calculate VAS once
            # For AD, we DON'T want the model to adapt to the fault order
            scan_order = self.fixed_scan_order
            
        # Reorder variables
        x = x[:, scan_order, :] # (B, M, L)
        
        # 3. Patching
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        N = patches.shape[2]
        
        # 4. Embedding
        enc_out = self.patch_embedding(patches)
        
        # 5. Variable Scan along Time (VST)
        enc_out = enc_out.reshape(B, M * N, self.d_model)
        
        # 6. Pass through TMB layers
        for layer in self.tmb_layers:
            enc_out = layer(enc_out)
            
        # 7. Output Head
        enc_out = enc_out.reshape(B, M, N, self.d_model)
        enc_out = enc_out.reshape(B, M, -1)
        enc_out = self.dropout(enc_out)
        out = self.head(enc_out)
        
        # Revert variable order
        inv_scan_order = torch.argsort(scan_order)
        out = out[:, inv_scan_order, :]
        
        # 8. RevIN Denorm
        out = self.revin(out, mode='denorm')
        
        return out
