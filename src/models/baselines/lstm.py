import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, horizon=512, dropout=0.2):
        """
        Mô hình Baseline LSTM cho bài toán Dự báo Chuỗi thời gian (Forecasting).
        
        Args:
            input_dim (int): Số lượng kênh cảm biến đầu vào (ví dụ: 2 cho B02).
            hidden_dim (int): Số lượng features ẩn trong LSTM.
            num_layers (int): Số lớp LSTM xếp chồng.
            horizon (int): Số bước thời gian (time steps) cần dự báo trong tương lai.
            dropout (float): Tỷ lệ Dropout để tránh overfitting.
        """
        super(LSTMForecaster, self).__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # Mạng LSTM xử lý chuỗi đầu vào (lookback)
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        
        # Thêm LayerNorm và Dropout trước khi dự báo để ổn định
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Ánh xạ trạng thái ẩn cuối cùng sang chuỗi tương lai
        self.fc = nn.Linear(hidden_dim, horizon * input_dim)
        
        # Khởi tạo trọng số
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
                
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        """
        Args:
            x: Tensor kích thước (batch_size, input_dim, lookback)
        Returns:
            pred: Tensor kích thước (batch_size, input_dim, horizon)
        """
        # LSTM yêu cầu input: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # out shape: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x)
        
        # Lấy trạng thái ở bước thời gian cuối cùng của lookback window
        last_out = out[:, -1, :] # (batch_size, hidden_dim)
        
        # Chuẩn hóa và Dropout
        last_out = self.ln(last_out)
        last_out = self.dropout(last_out)
        
        # Dự báo
        pred = self.fc(last_out) # (batch_size, horizon * input_dim)
        
        # Reshape về đúng kích thước Y: (batch_size, input_dim, horizon)
        pred = pred.view(-1, self.input_dim, self.horizon)
        return pred
