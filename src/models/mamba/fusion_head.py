import torch
import torch.nn as nn

class FusionForecastHead(nn.Module):
    def __init__(self, d_model=128, forecast_len=64, out_channels=1):
        """
        Giai đoạn 3: Dung hợp và trực tiếp xuất ra dự báo tương lai.
        """
        super().__init__()
        self.out_channels = out_channels
        self.forecast_len = forecast_len
        # Dự báo cho tất cả các kênh cùng lúc
        self.projection = nn.Linear(d_model, forecast_len * out_channels)

    def forward(self, x):
        # x: (Batch, Num_Patches, d_model)
        
        # Thay vì chỉ lấy token cuối, ta Pooling toàn bộ chuỗi các patch
        # x.transpose(1, 2) -> (Batch, d_model, Num_Patches)
        x_trans = x.transpose(1, 2)
        
        # Kết hợp Average Pooling (xu hướng chung) và Max Pooling (đỉnh bất thường)
        avg_pool = torch.mean(x_trans, dim=-1) # (Batch, d_model)
        max_pool, _ = torch.max(x_trans, dim=-1) # (Batch, d_model)
        
        combined = avg_pool + max_pool # (Batch, d_model)
        
        forecast = self.projection(combined) # (Batch, forecast_len * out_channels)
        
        # Reshape về (Batch, out_channels, forecast_len)
        if self.out_channels > 1:
            forecast = forecast.view(-1, self.out_channels, self.forecast_len)
        
        return forecast
