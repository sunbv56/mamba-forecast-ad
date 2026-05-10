import torch
import torch.nn as nn
from .layers import SimplePatchEmbedding
from .mamba_encoder import MambaEncoder
from .fusion_head import FusionForecastHead

class HybridMambaCNN(nn.Module):
    def __init__(self, config):
        """
        CI-Mamba: Channel-Independent Mamba.
        Xử lý từng kênh độc lập qua backbone Mamba chung.
        """
        super().__init__()
        model_cfg = config['model']
        data_cfg = config.get('data', {})
        
        in_channels = model_cfg.get('in_channels', data_cfg.get('input_dim', 1))
        out_channels = model_cfg.get('out_channels', data_cfg.get('input_dim', 1))
        patch_size = model_cfg.get('patch_size', data_cfg.get('patch_size', 64))
        stride = model_cfg.get('stride', data_cfg.get('stride', 64))
        
        # 1. Patching (Channel-Independent)
        self.patching = SimplePatchEmbedding(
            patch_size=patch_size,
            stride=stride,
            embed_dim=model_cfg['mamba_d_model']
        )
        
        # 2. Mamba Backbone (Shared across channels)
        # Tự động ánh xạ các tham số từ config cũ sang tham số MambaEncoder
        mamba_kwargs = model_cfg.get('mamba_kwargs', {}).copy()
        if 'mamba_d_state' in model_cfg: mamba_kwargs.setdefault('d_state', model_cfg['mamba_d_state'])
        if 'mamba_d_conv' in model_cfg: mamba_kwargs.setdefault('d_conv', model_cfg['mamba_d_conv'])
        if 'mamba_expand' in model_cfg: mamba_kwargs.setdefault('expand', model_cfg['mamba_expand'])
        
        self.mamba = MambaEncoder(
            d_model=model_cfg['mamba_d_model'],
            n_layer=model_cfg['mamba_n_layer'],
            version=model_cfg.get('mamba_version', 1),
            bidirectional=model_cfg.get('bidirectional', False), # Default to False for speed
            **mamba_kwargs
        )
        
        # 3. Forecasting Head
        self.head = FusionForecastHead(
            d_model=model_cfg['mamba_d_model'],
            forecast_len=model_cfg['forecast_len'],
            out_channels=1 # In CI mode, each channel is processed to output its own forecast
        )

    def forward(self, x, stats=None):
        # x: (Batch, Channels, Length)
        B, C, L = x.shape
        
        # Bước 1: Patching
        # x -> (Batch, Channels, Num_Patches, d_model)
        x = self.patching(x)
        B, C, N, D = x.shape
        
        # Bước 2: Backbone (CI - reshape Channels into Batch)
        x = x.reshape(B * C, N, D)
        x = self.mamba(x) # (B*C, N, D)
        
        # Xử lý stats cho CI mode
        if stats is not None:
            # stats: (B, C, 8) -> (B*C, 8)
            stats = stats.reshape(B * C, -1)
        
        # Bước 3: Head
        forecast = self.head(x, stats=stats) # (B*C, out_channels_per_head, forecast_len)
        
        # Reshape forecast back
        # forecast: (B*C, 1, forecast_len) -> (B, C, forecast_len)
        forecast = forecast.reshape(B, C, -1)
        
        return forecast
