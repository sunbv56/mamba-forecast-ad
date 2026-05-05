import torch
import torch.nn as nn

class CNNPatchEmbedding(nn.Module):
    def __init__(self, in_channels=2, patch_size=64, stride=64, embed_dim=128):
        """
        Giai đoạn 1: Trích xuất đặc trưng đa quy mô (Multi-scale) và nén chuỗi.
        Sử dụng các kernel khác nhau để bắt được cả đặc trưng thô và mịn.
        """
        super().__init__()
        
        # Đảm bảo embed_dim chia hết cho 2 để phân bổ cho 2 quy mô (hoặc dùng cat)
        mid_dim = embed_dim // 2
        
        # Quy mô 1: Kernel chuẩn
        self.conv1 = nn.Conv1d(in_channels, mid_dim, kernel_size=patch_size, stride=stride)
        
        # Quy mô 2: Kernel nhỏ hơn kết hợp với Dilated để giữ receptive field
        # Giúp bắt các biến động nhanh trong tín hiệu rung động
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels, mid_dim, kernel_size=patch_size // 2, stride=stride // 2),
            nn.MaxPool1d(kernel_size=2, stride=2), # Giữ nguyên output size để cộng/nối
            nn.BatchNorm1d(mid_dim)
        )
        
        self.norm = nn.BatchNorm1d(embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (Batch, Channels, Length)
        
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        
        # Nối đặc trưng từ 2 quy mô
        x = torch.cat([out1, out2], dim=1)
        
        x = self.norm(x)
        x = self.activation(x)
        
        # return: (Batch, Num_Patches, Embed_dim)
        return x.transpose(1, 2)
