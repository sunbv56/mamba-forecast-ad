import torch.nn as nn

class SelfSupervisedForecastingLoss(nn.Module):
    def __init__(self):
        """
        Hàm loss tự giám sát (MSE Loss) đo lường sai số dự báo.
        """
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)
