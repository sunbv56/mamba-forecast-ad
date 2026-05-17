import torch
import torch.nn as nn
from .layers import RevIN, SeriesDecomposition, MultiScalePatchEmbedding, SimplePatchEmbedding
from .mamba_encoder import MambaEncoder
from .fusion_head import FusionForecastHead


class HybridMambaCNN(nn.Module):
    """
    CI-Mamba++ (Channel-Independent Mamba with Decomposition & RevIN).

    Kiến trúc:
        [Input] (B, C, L)
            │
        [RevIN]   ← chuẩn hóa, lưu mean/std để denorm đầu ra
            │
        [Series Decomposition]
           ┌───────────────┐
      [Seasonal branch]  [Trend branch]
    MultiScalePatch       Linear(L → H)
         │
     MambaEncoder (CI)
         │
     FusionHead
           └───────────────┘
               [Add]
        [RevIN denorm]
            │
        [Output] (B, C, H)

    Ưu điểm so với phiên bản cũ:
    - RevIN: ổn định training khi dữ liệu non-stationary (bearing degradation).
    - Series Decomp: Trend & Seasonal được xử lý tách biệt → Mamba chỉ lo phần
      dao động nhanh, Linear lo xu hướng → hội tụ nhanh hơn nhiều.
    - MultiScalePatchEmbedding: bắt được đặc trưng tần số cao, vừa và thấp.
    - Stats head (FusionHead) vẫn giữ nguyên để tương thích với pipeline hiện tại.
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config['model']
        data_cfg  = config.get('data', {})

        # --- Hyper-parameters ---
        in_channels  = model_cfg.get('in_channels',  data_cfg.get('input_dim', 1))
        patch_size   = model_cfg.get('patch_size',   data_cfg.get('patch_size', 64))
        stride       = model_cfg.get('stride',       data_cfg.get('stride', 32))
        d_model      = model_cfg['mamba_d_model']
        forecast_len = model_cfg['forecast_len']
        lookback     = model_cfg.get('lookback', data_cfg.get('lookback', 1024))
        decomp_kernel = model_cfg.get('decomp_kernel', 25)

        # ------------------------------------------------------------------
        # 1. RevIN — học được affine per-channel
        # ------------------------------------------------------------------
        self.revin = RevIN(num_features=in_channels, affine=False)

        # ------------------------------------------------------------------
        # 2. Series Decomposition
        # ------------------------------------------------------------------
        self.decomp = SeriesDecomposition(kernel_size=decomp_kernel)

        # ------------------------------------------------------------------
        # 3a. Seasonal Branch — Multi-scale Patch + Mamba
        # ------------------------------------------------------------------
        use_multiscale = model_cfg.get('use_multiscale', True)
        if use_multiscale:
            self.patching = MultiScalePatchEmbedding(
                patch_size=patch_size,
                stride=stride,
                embed_dim=d_model
            )
        else:
            self.patching = SimplePatchEmbedding(
                patch_size=patch_size,
                stride=stride,
                embed_dim=d_model
            )

        # Mamba backbone (Channel-Independent: channels folded into batch dim)
        mamba_kwargs = model_cfg.get('mamba_kwargs', {}).copy()
        if 'mamba_d_state' in model_cfg: mamba_kwargs.setdefault('d_state', model_cfg['mamba_d_state'])
        if 'mamba_d_conv'  in model_cfg: mamba_kwargs.setdefault('d_conv',  model_cfg['mamba_d_conv'])
        if 'mamba_expand'  in model_cfg: mamba_kwargs.setdefault('expand',  model_cfg['mamba_expand'])

        self.mamba = MambaEncoder(
            d_model=d_model,
            n_layer=model_cfg['mamba_n_layer'],
            version=model_cfg.get('mamba_version', 1),
            bidirectional=model_cfg.get('bidirectional', False),
            **mamba_kwargs
        )

        # Forecasting head for seasonal branch (CI mode → out_channels=1 per head)
        self.seasonal_head = FusionForecastHead(
            d_model=d_model,
            forecast_len=forecast_len,
            out_channels=1
        )

        # ------------------------------------------------------------------
        # 3b. Trend Branch — simple Linear (DLinear-style) with optional Pooling
        # ------------------------------------------------------------------
        self.trend_downsample = model_cfg.get('trend_downsample', 1)
        if self.trend_downsample > 1:
            self.trend_pool = nn.AvgPool1d(kernel_size=self.trend_downsample)
            self.trend_head = nn.Linear(lookback // self.trend_downsample, forecast_len)
        else:
            self.trend_head = nn.Linear(lookback, forecast_len)

        # ------------------------------------------------------------------
        # 4. Learnable mix weight (α·seasonal + (1-α)·trend per channel)
        # ------------------------------------------------------------------
        self.mix_alpha = nn.Parameter(torch.full((in_channels,), 0.5))

    def forward(self, x, stats=None):
        """
        x     : (Batch, Channels, Length)
        stats : (Batch, Channels, 8)   [optional physical features]
        →       (Batch, Channels, forecast_len)
        """
        B, C, L = x.shape

        # --- RevIN normalize ---
        x = self.revin(x, mode='norm')                # (B, C, L)

        # --- Series Decomposition ---
        seasonal, trend = self.decomp(x)              # both (B, C, L)

        # ── Trend Branch ──────────────────────────────────────────────────
        if self.trend_downsample > 1:
            trend_pooled = self.trend_pool(trend)      # (B, C, L // downsample)
            trend_out = self.trend_head(trend_pooled)  # (B, C, forecast_len)
        else:
            trend_out = self.trend_head(trend)         # (B, C, forecast_len)

        # ── Seasonal Branch ───────────────────────────────────────────────
        # Patching: (B, C, L) → (B, C, N, d_model)
        s = self.patching(seasonal)                    # (B, C, N, D)
        _, _, N, D = s.shape

        # Channel-Independent: fold C into batch
        s = s.reshape(B * C, N, D)                    # (B*C, N, D)
        s = self.mamba(s)                              # (B*C, N, D)

        # Stats for CI mode
        stats_ci = None
        if stats is not None:
            stats_ci = stats.reshape(B * C, -1)       # (B*C, 8)

        s_out = self.seasonal_head(s, stats=stats_ci)  # (B*C, 1, forecast_len)
        s_out = s_out.reshape(B, C, -1)                # (B, C, forecast_len)

        # ── Learnable Mixing ──────────────────────────────────────────────
        alpha = torch.sigmoid(self.mix_alpha).view(1, C, 1)   # (1, C, 1)
        forecast = alpha * s_out + (1.0 - alpha) * trend_out  # (B, C, forecast_len)

        # --- RevIN de-normalize ---
        forecast = self.revin(forecast, mode='denorm')  # (B, C, forecast_len)

        return forecast

