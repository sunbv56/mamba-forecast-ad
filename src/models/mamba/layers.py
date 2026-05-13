import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN).
    Ref: Kim et al. (2021) - "Reversible Instance Normalization for Accurate
    Time-Series Forecasting against Distribution Shift"

    Giải quyết vấn đề Non-stationarity: chuẩn hóa đầu vào và "giải" lại đầu ra,
    giúp Mamba học trên dữ liệu rung động đang thay đổi phân phối liên tục.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        """
        x    : (Batch, Channels, Length)
        mode : 'norm' để chuẩn hóa, 'denorm' để giải chuẩn hóa
        """
        if mode == 'norm':
            # Tính thống kê theo chiều thời gian (Length)
            self._mean = x.mean(dim=-1, keepdim=True)          # (B, C, 1)
            self._std  = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)  # (B, C, 1)
            x = (x - self._mean) / self._std
            if self.affine:
                # affine_weight / bias có shape (C,) → reshape để broadcast
                x = x * self.affine_weight.view(1, -1, 1) + self.affine_bias.view(1, -1, 1)
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias.view(1, -1, 1)) / (self.affine_weight.view(1, -1, 1) + self.eps)
            x = x * self._std + self._mean
        return x


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block: tách Trend và Seasonal/Residual.
    Kernel lớn → Trend chậm (xu hướng mòn); Seasonal → dao động nhanh.
    Inspired by DLinear / Autoformer.
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        # padding='same' equivalent: padding = kernel_size // 2
        padding = kernel_size // 2
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        """
        x      : (Batch, Channels, Length)
        returns: seasonal (B, C, L), trend (B, C, L)
        """
        trend = self.moving_avg(x)
        if trend.shape[-1] != x.shape[-1]:
            trend = nn.functional.interpolate(
                trend, size=x.shape[-1], mode='linear', align_corners=False
            )
        seasonal = x - trend
        return seasonal, trend


class MultiScalePatchEmbedding(nn.Module):
    """
    Multi-Scale CNN Patch Embedding cho tín hiệu rung động.
    Dùng 3 CNN kernel khác nhau để bắt đặc trưng:
    - Fine   (kernel nhỏ):  biến động cục bộ, rung động tần số cao
    - Medium (kernel vừa):  pattern chu kỳ ngắn
    - Coarse (kernel lớn):  xu hướng chậm, vòng quay

    Output: (Batch, Channels, Num_Patches, embed_dim)  – giống SimplePatchEmbedding
    để thay thế trực tiếp trong HybridMambaCNN.
    """
    def __init__(self, patch_size: int = 64, stride: int = 32, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.stride     = stride
        self.embed_dim  = embed_dim

        # Chia embed_dim thành 3 phần (bổ sung phần dư vào coarse)
        d_fine   = embed_dim // 4       # 25%
        d_medium = embed_dim // 4       # 25%
        d_coarse = embed_dim - d_fine - d_medium  # 50%

        self.proj_fine   = nn.Linear(patch_size // 4 if patch_size // 4 >= 1 else 1, d_fine)
        self.proj_medium = nn.Linear(patch_size // 2 if patch_size // 2 >= 1 else 1, d_medium)
        self.proj_coarse = nn.Linear(patch_size,      d_coarse)

        self.norm = nn.LayerNorm(embed_dim)

    def _unfold(self, x: torch.Tensor, size: int, step: int) -> torch.Tensor:
        """Unfold (B*C, L) → (B*C, N, size); clamp step to avoid 0."""
        step = max(step, 1)
        patches = x.unfold(dimension=-1, size=size, step=step)   # (B*C, N, size)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (Batch, Channels, Length)
        → (Batch, Channels, Num_Patches, embed_dim)
        """
        B, C, L = x.shape
        x_flat = x.reshape(B * C, L)   # (B*C, L)

        ps = self.patch_size
        st = self.stride

        # --- Coarse patches (full patch_size) ---
        patches_c = self._unfold(x_flat, ps, st)           # (B*C, N, ps)
        N = patches_c.shape[1]
        out_c = self.proj_coarse(patches_c)                # (B*C, N, d_coarse)

        # --- Medium patches (patch_size//2, same N) ---
        ps_m = max(ps // 2, 1)
        st_m = max(st // 2, 1)
        patches_m = self._unfold(x_flat, ps_m, st_m)
        # Có thể N_m ≠ N → resize
        if patches_m.shape[1] != N:
            patches_m = patches_m[:, :N, :] if patches_m.shape[1] > N else \
                torch.nn.functional.pad(patches_m, (0, 0, 0, N - patches_m.shape[1]))
        out_m = self.proj_medium(patches_m)                # (B*C, N, d_medium)

        # --- Fine patches (patch_size//4, same N) ---
        ps_f = max(ps // 4, 1)
        st_f = max(st // 4, 1)
        patches_f = self._unfold(x_flat, ps_f, st_f)
        if patches_f.shape[1] != N:
            patches_f = patches_f[:, :N, :] if patches_f.shape[1] > N else \
                torch.nn.functional.pad(patches_f, (0, 0, 0, N - patches_f.shape[1]))
        out_f = self.proj_fine(patches_f)                  # (B*C, N, d_fine)

        # --- Concat & Norm ---
        out = torch.cat([out_f, out_m, out_c], dim=-1)    # (B*C, N, embed_dim)
        out = self.norm(out)
        out = out.reshape(B, C, N, self.embed_dim)         # (B, C, N, embed_dim)
        return out


class SimplePatchEmbedding(nn.Module):
    """
    Simple Linear Patching (Standard for PatchTST/iTransformer).
    Giữ lại để tương thích ngược.
    """
    def __init__(self, patch_size=64, stride=64, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        B, C, L = x.shape
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        B, C, N, P = patches.shape
        patches = patches.reshape(-1, P)
        x = self.projection(patches)
        x = x.reshape(B, C, N, -1)
        return x  # (B, C, N, d_model)

