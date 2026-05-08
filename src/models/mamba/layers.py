import torch
import torch.nn as nn

class SeriesDecomposition(nn.Module):
    """
    Series decomposition block to separate trend and seasonal/residual components.
    Inspired by DLinear and Autoformer.
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        """
        x: (Batch, Channels, Length)
        returns: (seasonal, trend)
        """
        # Padding to keep length same (if not handled by AvgPool1d padding correctly for even kernels)
        # AvgPool1d with padding=kernel_size//2 usually works for odd kernels.
        
        trend = self.moving_avg(x)
        # If kernel_size is even, AvgPool1d padding might make it L+1 or L-1. 
        # We ensure it matches x's length.
        if trend.shape[-1] != x.shape[-1]:
            trend = nn.functional.interpolate(trend, size=x.shape[-1], mode='linear', align_corners=False)
            
        seasonal = x - trend
        return seasonal, trend

class SimplePatchEmbedding(nn.Module):
    """
    Simple Linear Patching (Standard for PatchTST/iTransformer).
    Faster and more stable than CNN for many time-series tasks.
    """
    def __init__(self, patch_size=64, stride=64, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        # x: (Batch, 1, Length) - processing one channel at a time
        B, C, L = x.shape
        
        # Unfold to create patches
        # (Batch, 1, Num_Patches, Patch_Size)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        B, C, N, P = patches.shape
        
        # Reshape for projection
        patches = patches.reshape(-1, P) # (B*C*N, P)
        x = self.projection(patches) # (B*C*N, d_model)
        x = x.reshape(B, C, N, -1) # (B, C, N, d_model)
        
        return x # (B, C, N, d_model)
