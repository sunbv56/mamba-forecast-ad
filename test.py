import torch
import torch.nn as nn
import sys

# Check for CUDA (Official Mamba REQUIRES CUDA)
if not torch.cuda.is_available():
    print("\n[LỖI] Mamba chính thức yêu cầu GPU NVIDIA và CUDA.")
    print("Vui lòng chạy trong môi trường có hỗ trợ GPU (như WSL2 với NVIDIA Drivers).")
    sys.exit(1)

try:
    from mamba_ssm import Mamba
except ImportError:
    print("\n[LỖI] Không tìm thấy thư viện mamba_ssm.")
    print("Vui lòng chạy lệnh: pip install mamba-ssm causal-conv1d>=1.4.0")
    sys.exit(1)

# ==========================================
# ĐỊNH NGHĨA MÔ HÌNH CHUỖI THỜI GIAN (OFFICIAL)
# ==========================================
class MambaTimeSeriesForecaster(nn.Module):
    def __init__(self, num_channels, seq_len, patch_size, d_model, forecast_len, num_mamba_layers=2):
        """
        Mô hình dự báo chuỗi thời gian sử dụng thư viện Mamba chính thức (state-spaces).
        Tối ưu hóa cực cao cho GPU NVIDIA.
        """
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.forecast_len = forecast_len
        
        # Tính số lượng patch
        assert seq_len % patch_size == 0, "Độ dài chuỗi (seq_len) phải chia hết cho kích thước patch (patch_size)"
        self.num_patches = seq_len // patch_size
        
        # --- GIAI ĐOẠN 1: Patching & Projection ---
        # Chuyển đổi các patch dữ liệu thành vector đặc trưng (tokens)
        self.projection = nn.Linear(num_channels * patch_size, d_model)
        
        # --- GIAI ĐOẠN 2: Khối Mamba Official ---
        # Sử dụng nn.ModuleList để xếp chồng các lớp Mamba chính thức
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,    # Model dimension
                d_state=16,         # SSM state expansion factor
                d_conv=4,           # Local convolution width
                expand=2,           # Block expansion factor
            ) for _ in range(num_mamba_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        
        # --- GIAI ĐOẠN 3: Forecast Head ---
        # Lớp tuyến tính cuối cùng để dự báo tương lai
        self.head = nn.Linear(self.num_patches * d_model, num_channels * forecast_len)

    def forward(self, x):
        """
        Đầu vào x: (Batch_Size, Num_Channels, Sequence_Length)
        """
        B, C, L = x.shape
        
        # 1. Thực hiện Patching
        # Gộp các điểm thời gian liên tiếp thành các "patch" để giảm độ dài chuỗi tokens
        x_patched = x.view(B, C, self.num_patches, self.patch_size)
        x_patched = x_patched.permute(0, 2, 1, 3).contiguous()
        x_flattened = x_patched.view(B, self.num_patches, C * self.patch_size)
        
        # 2. Tạo Tokens (B, Num_Patches, d_model)
        x_tokens = self.projection(x_flattened)
        
        # 3. Chạy qua các lớp Mamba chính thức (Optimized CUDA Kernels)
        x_mamba = x_tokens
        for layer in self.mamba_layers:
            x_mamba = layer(x_mamba)
        
        x_mamba = self.norm_f(x_mamba)
        
        # 4. Đưa vào lớp dự báo
        x_mamba_flat = x_mamba.view(B, -1)
        out = self.head(x_mamba_flat)
        
        # 5. Định dạng lại đầu ra (Batch_Size, Num_Channels, Forecast_Len)
        out = out.view(B, C, self.forecast_len)
        
        return out, x_tokens

# ==========================================
# CHẠY THỬ NGHIỆM VỚI DỮ LIỆU GIẢ
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda") # Bắt buộc dùng CUDA cho bản Official
    print(f"\n--- Đang chạy trên thiết bị: {torch.cuda.get_device_name(0)} ---")

    # Siêu tham số
    BATCH_SIZE = 16
    NUM_CHANNELS = 5      
    SEQ_LEN = 512         
    PATCH_SIZE = 16       
    D_MODEL = 64         
    FORECAST_LEN = 96     
    NUM_MAMBA_LAYERS = 2  

    model = MambaTimeSeriesForecaster(
        num_channels=NUM_CHANNELS,
        seq_len=SEQ_LEN,
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        forecast_len=FORECAST_LEN,
        num_mamba_layers=NUM_MAMBA_LAYERS
    ).to(device)

    # Sinh dữ liệu ngẫu nhiên (Batch, Cảm biến, Thời gian)
    dummy_input = torch.randn(BATCH_SIZE, NUM_CHANNELS, SEQ_LEN).to(device)

    print("\n[Trạng thái Tensor]")
    print(f"1. Dữ liệu lịch sử (Input)     : {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        forecast_output, embed_tokens = model(dummy_input)
        
    print(f"2. Token sau khi Patching      : {embed_tokens.shape}")
    print(f"3. Kết quả dự báo (Output)     : {forecast_output.shape}")
    
    print("\n✅ THÀNH CÔNG! Đã chạy Mamba Official với CUDA Kernels tối ưu.")
