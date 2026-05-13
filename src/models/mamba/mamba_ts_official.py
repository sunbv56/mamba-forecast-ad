import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional

from .layers import RevIN
from .layers_official.mamba_ssm.mixer2_seq_simple import MixerTSModel

class MambaTSConfig:
    """
    Configuration object for the official MambaTS model.
    """
    def __init__(self, 
                 in_channels: int, 
                 lookback: int, 
                 forecast_len: int, 
                 patch_size: int = 64, 
                 stride: int = 32, 
                 d_model: int = 128, 
                 n_layers: int = 4, 
                 dropout: float = 0.1,
                 VPT_mode: int = 1, # 1 for Variable-Aware Scanning
                 ATSP_solver: str = 'SA', # 'SA' (Simulated Annealing)
                 ssm_cfg: dict = None,
                 use_casual_conv: bool = False):
        self.task_name = 'forecasting'
        self.enc_in = in_channels
        self.seq_len = lookback
        self.pred_len = forecast_len
        self.patch_len = patch_size
        self.stride = stride
        self.d_model = d_model
        self.e_layers = n_layers
        self.dropout = dropout
        self.VPT_mode = VPT_mode
        self.ATSP_solver = ATSP_solver
        self.ssm_cfg = ssm_cfg if ssm_cfg else {'layer': 'Mamba1'}
        self.use_casual_conv = use_casual_conv

class PredictionHead(nn.Module):
    def __init__(self, in_nf, out_nf):
        super().__init__()
        self.linear = nn.Linear(in_nf, out_nf)

    def forward(self, x, n_vars):
        # x: (B, M*N, D)
        # x: (B, N*C, D) -> (B*C, N, D)
        x = rearrange(x, 'b (n c) d -> (b c) n d', c=n_vars)
        # Flatten patches for each variable: (B*C, N*D)
        x = rearrange(x, 'b n d -> b (n d)')
        x = self.linear(x)
        x = rearrange(x, '(b c) p -> b c p', c=n_vars)
        return x

class MambaTSOfficial(nn.Module):
    """
    Official Implementation of MambaTS (arXiv:2405.16440)
    Integrated with project's RevIN and standard training pipeline.
    """
    def __init__(self, configs: MambaTSConfig):
        super().__init__()
        self.configs = configs
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.n_vars = configs.enc_in
        self.pred_len = configs.pred_len

        # 1. RevIN for Standardization
        # Set affine=False for Anomaly Detection to prevent the model from 
        # learning to normalize away the anomaly signal scale.
        self.revin = RevIN(num_features=configs.enc_in, affine=False)

        # 2. Patching & Embedding
        # The official model uses a simple linear projection for patches
        self.value_embedding = nn.Linear(configs.patch_len, configs.d_model, bias=False)

        # 3. Encoder (Official MixerTSModel)
        self.encoder = MixerTSModel(
            d_model=configs.d_model,
            n_layer=configs.e_layers,
            n_vars=configs.enc_in,
            dropout=configs.dropout,
            ssm_cfg=configs.ssm_cfg,
            VPT_mode=configs.VPT_mode,
            ATSP_solver=configs.ATSP_solver,
            use_casual_conv=configs.use_casual_conv,
        )

        # 4. Prediction Head
        # num_patches = (seq_len - patch_len) // stride + 1
        num_patches = (configs.seq_len - configs.patch_len) // configs.stride + 1
        self.head = PredictionHead(configs.d_model * num_patches, configs.pred_len)

        # 5. Operating Conditions Embedding (Optional)
        # B02 has 6 OC features: setDynLoad, peak_dynLoad, setStatLoad, meanAbs_statLoad, setSpeed, meanAbs_speed
        self.oc_embedding = nn.Linear(6, configs.d_model)

    def forward(self, x: torch.Tensor, oc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, M, L) - M variables, L lookback
        oc: (B, 6) - Operating conditions (Speed, Load, etc.)
        """
        B, M, L = x.shape
        
        # 1. RevIN Norm
        x = self.revin(x, mode='norm') # (B, M, L)

        # 2. Patching
        # x: (B, M, L) -> unfold -> (B, M, N, P)
        x_patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        N = x_patches.shape[2]
        
        # 3. Embedding
        # (B, M, N, P) -> (B*M, N, P) -> Linear -> (B*M, N, D)
        x_emb = self.value_embedding(x_patches.reshape(B * M, N, -1))
        
        # 4. Inject Operating Conditions (OC)
        if oc is not None:
            # oc: (B, 6) -> embed -> (B, D)
            oc_emb = self.oc_embedding(oc) # (B, D)
            # Expand to (B, M, N, D) and then reshape to (B*M, N, D)
            oc_emb = oc_emb.unsqueeze(1).unsqueeze(1).expand(B, M, N, -1)
            x_emb = x_emb + oc_emb.reshape(B * M, N, -1)

        # 5. Variable Scan along Time (VST) - Alternating Fashion (Paper proposal)
        # Combine variables and patches: (B, N*M, D) - order: P1_V1, P1_V2, P2_V1...
        # x_emb: (B*M, N, D) -> rearrange -> (B, N*M, D)
        enc_in = rearrange(x_emb, '(b c) n d -> b (n c) d', b=B, c=M)

        # 6. Encoder
        # enc_out: (B, M*N, D)
        enc_out, _ = self.encoder(enc_in)

        # 7. Decoder / Head
        # (B, M*N, D) -> (B, M, pred_len)
        dec_out = self.head(enc_out, n_vars=M) # (B, M, pred_len)

        # 8. RevIN Denorm
        # dec_out: (B, M, pred_len)
        dec_out = self.revin(dec_out, mode='denorm')
        
        return dec_out

    def batch_update_state(self, cost_tensor):
        """
        Update the adjacency matrix for VAST based on training loss/cost.
        cost_tensor: (B, M, L) or similar loss-related tensor
        """
        self.encoder.batch_update_state(cost_tensor)
