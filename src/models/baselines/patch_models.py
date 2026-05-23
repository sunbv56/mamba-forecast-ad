import torch
import torch.nn as nn
from src.models.mamba.cnn_patching import CNNPatchEmbedding
from src.models.mamba.fusion_head import FusionForecastHead
from src.models.baselines.transformer_small import PositionalEncoding

class PatchTST(nn.Module):
    """
    PatchTST (Patch Time Series Transformer) - Channel-Independent Patch-based Transformer.
    Segments time series into patches and applies Transformer self-attention over patch tokens in a Channel-Independent fashion.
    """
    def __init__(self, in_channels=2, lookback=1024, patch_size=64, stride=64, 
                 d_model=64, nhead=4, num_layers=3, dropout=0.1, horizon=512, head_type='pooling'):
        super().__init__()
        self.in_channels = in_channels
        self.horizon = horizon
        self.lookback = lookback
        self.patch_size = patch_size
        self.stride = stride
        self.head_type = head_type
        
        # Calculate number of patches: N = (L - P) // S + 1
        self.num_patches = (lookback - patch_size) // stride + 1
        
        # Patch projection: maps a 1D patch of size patch_size to d_model
        self.embedding = nn.Linear(patch_size, d_model)
        
        # Positional Encoding for patch sequence
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 10)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Forecasting Head
        if self.head_type == 'pooling':
            self.head = nn.Linear(d_model, horizon)
        else:
            self.head = nn.Linear(d_model * self.num_patches, horizon)
        
    def forward(self, x):
        # x: (B, C, L) where B=batch_size, C=in_channels, L=lookback
        B, C, L = x.shape
        
        # 1. Channel Independence: Treat each channel as a separate sample
        # Shape: (B * C, L)
        x = x.reshape(B * C, L)
        
        # 2. Patching: (B * C, L) -> (B * C, num_patches, patch_size)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        
        # 3. Embedding & Positional Encoding
        x = self.embedding(x) # (B * C, num_patches, d_model)
        x = self.pos_encoder(x)
        
        # 4. Transformer over patches
        x = self.transformer(x) # (B * C, num_patches, d_model)
        
        # 5. Head
        if self.head_type == 'pooling':
            x = x.mean(dim=1) # (B * C, d_model)
        else:
            x = x.reshape(B * C, -1) # (B * C, num_patches * d_model)
        x = self.head(x) # (B * C, horizon)
        
        # 6. Reshape back to (B, C, horizon)
        x = x.reshape(B, C, self.horizon)
        
        return x

# Alias for backward compatibility
VanillaTransformer = PatchTST


class PatchLSTM(nn.Module):
    def __init__(self, in_channels=2, patch_size=64, stride=64, d_model=96, num_layers=2, horizon=512):
        super().__init__()
        self.patching = CNNPatchEmbedding(in_channels, patch_size, stride, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.head = FusionForecastHead(d_model, horizon, out_channels=in_channels, use_stats=False)

    def forward(self, x):
        x = self.patching(x)
        x, _ = self.lstm(x)
        return self.head(x)
