import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Standard Temporal Block from Bai et al. (2018)
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNForecaster(nn.Module):
    """
    TCN Forecaster based on 'An Empirical Evaluation of Generic Convolutional and Recurrent 
    Networks for Sequence Modeling' (Bai et al., 2018).
    
    Improved with a forecasting head that uses more than just the last step.
    """
    def __init__(self, input_dim=2, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2, horizon=512):
        super(TCNForecaster, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        
        # Head: Global Pooling for parameter efficiency
        self.fc = nn.Linear(num_channels[-1], horizon * input_dim)
        
        self.horizon = horizon
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (batch, input_dim, lookback)
        y1 = self.tcn(x) # (batch, num_channels[-1], lookback)
        
        # Global Average Pooling + Global Max Pooling
        avg_pool = torch.mean(y1, dim=-1)
        max_pool, _ = torch.max(y1, dim=-1)
        out = avg_pool + max_pool # (batch, num_channels[-1])
        
        out = self.fc(out)
        out = out.view(-1, self.input_dim, self.horizon)
        return out
