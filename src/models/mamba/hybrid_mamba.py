import torch
import torch.nn as nn
from .cnn_patching import CNNPatchEmbedding
from .mamba_encoder import MambaEncoder
from .fusion_head import FusionForecastHead

class HybridMambaCNN(nn.Module):
    def __init__(self, config):
        """
        Mô hình tổng hợp theo kiến trúc đề xuất.
        """
        super().__init__()
        model_cfg = config['model']
        data_cfg = config.get('data', {})
        
        in_channels = model_cfg.get('in_channels', data_cfg.get('input_dim', 1))
        out_channels = model_cfg.get('out_channels', data_cfg.get('input_dim', 1))
        patch_size = model_cfg.get('patch_size', data_cfg.get('patch_size', 64))
        stride = model_cfg.get('stride', data_cfg.get('stride', 64))
        
        # Giai đoạn 1:
        self.patching = CNNPatchEmbedding(
            in_channels=in_channels, 
            patch_size=patch_size,
            stride=stride,
            embed_dim=model_cfg['cnn_out_channels']
        )
        
        # Lớp chiếu từ CNN dim sang Mamba dim (nếu khác nhau)
        self.project_to_mamba = nn.Linear(model_cfg['cnn_out_channels'], model_cfg['mamba_d_model'])
        self.norm = nn.LayerNorm(model_cfg['mamba_d_model'])
        
        # Giai đoạn 2:
        self.mamba = MambaEncoder(
            d_model=model_cfg['mamba_d_model'],
            n_layer=model_cfg['mamba_n_layer'],
            d_state=model_cfg['mamba_d_state'],
            d_conv=model_cfg['mamba_d_conv'],
            expand=model_cfg['mamba_expand']
        )
        
        # Giai đoạn 3:
        self.head = FusionForecastHead(
            d_model=model_cfg['mamba_d_model'],
            forecast_len=model_cfg['forecast_len'],
            out_channels=out_channels
        )

    def forward(self, x):
        # x: (Batch, 1, Sequence_Length)
        x = self.patching(x)                  # -> (Batch, Num_Patches, CNN_Dim)
        x = self.project_to_mamba(x)          # -> (Batch, Num_Patches, Mamba_Dim)
        x = self.norm(x)                      # -> (Batch, Num_Patches, Mamba_Dim)
        x = self.mamba(x)                     # -> (Batch, Num_Patches, Mamba_Dim)
        forecast = self.head(x)               # -> (Batch, forecast_len)
        return forecast
