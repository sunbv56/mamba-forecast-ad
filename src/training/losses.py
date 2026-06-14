import torch.nn as nn

class SelfSupervisedForecastingLoss(nn.Module):
    def __init__(self, loss_type='huber'):
        """
        Hàm loss tự giám sát đo lường sai số dự báo.
        Mặc định sử dụng Huber Loss.
        """
        super().__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.HuberLoss(delta=1.0)

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)
