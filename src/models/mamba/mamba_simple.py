import torch
import torch.nn as nn
from .layers import SimplePatchEmbedding
from .mamba_encoder import MambaEncoder

class SimpleMamba(nn.Module):
    """
    Simple Mamba model for time-series forecasting.
    Bypasses RevIN, decomposition, statistics head, and multiscale patching.
    Uses SimplePatchEmbedding + MambaEncoder (CI mode) + Linear Head.
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

        self.in_channels = in_channels
        self.forecast_len = forecast_len

        # 1. Simple Patch Embedding
        self.patching = SimplePatchEmbedding(
            patch_size=patch_size,
            stride=stride,
            embed_dim=d_model
        )

        # 2. Mamba Encoder (Channel-Independent)
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

        # 3. Simple Linear Head (CI mode -> out_channels=1 per head, then reshape back)
        self.head = nn.Linear(d_model, forecast_len)

    def forward(self, x, stats=None):
        """
        x : (Batch, Channels, Length)
        returns: (Batch, Channels, forecast_len)
        """
        B, C, L = x.shape

        # Patching: (Batch, Channels, Length) -> (Batch, Channels, Num_Patches, d_model)
        s = self.patching(x)
        _, _, N, D = s.shape

        # Channel-Independent: fold C into batch dimension
        s = s.reshape(B * C, N, D)                  # (B*C, N, D)
        s = self.mamba(s)                            # (B*C, N, D)

        # Global Average Pooling (GAP) over patches
        s = s.mean(dim=1)                            # (B*C, D)
        out = self.head(s)                           # (B*C, forecast_len)
        out = out.reshape(B, C, self.forecast_len)   # (B, C, forecast_len)

        return out
