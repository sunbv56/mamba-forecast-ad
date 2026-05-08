import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

try:
    from mamba_ssm import Mamba3
except ImportError:
    Mamba3 = None

class MambaEncoder(nn.Module):
    def __init__(self, d_model=128, n_layer=4, version=1, bidirectional=True, **kwargs):
        """
        Giai đoạn 2: Học tương quan dài hạn bằng kiến trúc Mamba (v1, v2, hoặc v3).
        Hỗ trợ Bidirectional Mamba để bắt ngữ cảnh hai chiều.
        
        Args:
            d_model: Kích thước model.
            n_layer: Số lớp Mamba.
            version: Phiên bản Mamba (1, 2, hoặc 3).
            bidirectional: Có sử dụng hai chiều hay không.
            **kwargs: Các tham số riêng cho từng phiên bản (d_state, d_conv, expand, headdim, etc.)
        """
        super().__init__()
        
        # Chọn lớp Mamba dựa trên phiên bản
        if version == 1:
            mamba_cls = Mamba
        elif version == 2:
            mamba_cls = Mamba2
        elif version == 3:
            mamba_cls = Mamba3
        else:
            raise ValueError(f"Phiên bản Mamba không hợp lệ: {version}")

        if mamba_cls is None:
            raise ImportError(f"Vui lòng cài đặt mamba_ssm để sử dụng Mamba v{version}")
            
        self.bidirectional = bidirectional
        self.n_layer = n_layer
        self.version = version

        # Cấu hình tham số thống nhất
        mamba_params = {"d_model": d_model}
        
        if version in [1, 2]:
            mamba_params.update({
                "d_state": kwargs.get("d_state", 16),
                "d_conv": kwargs.get("d_conv", 4),
                "expand": kwargs.get("expand", 2),
            })
        elif version == 3:
            mamba_params.update({
                "d_state": kwargs.get("d_state", 128),
                "headdim": kwargs.get("headdim", 64),
                "is_mimo": kwargs.get("is_mimo", True),
                "mimo_rank": kwargs.get("mimo_rank", 4),
                "chunk_size": kwargs.get("chunk_size", 16),
                "is_outproj_norm": kwargs.get("is_outproj_norm", False),
            })
            # Nếu người dùng truyền các tham số khác, ta vẫn cho phép ghi đè
            for k, v in kwargs.items():
                if k not in ["d_state", "d_model"]: # Tránh ghi đè d_model đã set
                    mamba_params[k] = v

        self.layers = nn.ModuleList([
            mamba_cls(**mamba_params) for _ in range(n_layer)
        ])
        
        if bidirectional:
            self.layers_rev = nn.ModuleList([
                mamba_cls(**mamba_params) for _ in range(n_layer)
            ])
            
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layer)
        ])

    def forward(self, x):
        # x: (Batch*Channels, Seq_Len, d_model)
        for i in range(self.n_layer):
            res = x
            x = self.norms[i](x)
            
            # Forward pass
            out = self.layers[i](x)
            
            if self.bidirectional:
                # Backward pass: reverse seq, process, reverse back
                x_rev = x.flip(dims=[1])
                out_rev = self.layers_rev[i](x_rev)
                out_rev = out_rev.flip(dims=[1])
                # Combine
                out = (out + out_rev) / 2
                
            x = res + out
                
        return x
