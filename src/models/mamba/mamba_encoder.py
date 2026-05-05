import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class MambaEncoder(nn.Module):
    def __init__(self, d_model=128, n_layer=4, d_state=16, d_conv=4, expand=2):
        """
        Giai đoạn 2: Học tương quan dài hạn bằng kiến trúc Mamba.
        Đã thêm LayerNorm và Residual Connection để ổn định huấn luyện.
        """
        super().__init__()
        if Mamba is None:
            raise ImportError("Vui lòng cài đặt mamba_ssm")
            
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layer)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layer)
        ])

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        for layer, norm in zip(self.layers, self.norms):
            # Sử dụng Pre-Norm và Residual Connection
            x = x + layer(norm(x))
        return x
