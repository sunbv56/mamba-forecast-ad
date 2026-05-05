import torch
import torch.nn.functional as F

def calculate_anomaly_score(y_true, y_pred, metric='mse'):
    """
    Tính Anomaly Score giữa chuỗi thực tế và chuỗi dự báo.
    y_true: Tensor (B, C, H)
    y_pred: Tensor (B, C, H)
    Trả về điểm số bất thường cho mỗi mẫu trong batch (B,)
    """
    if metric == 'mse':
        loss = F.mse_loss(y_pred, y_true, reduction='none')
    elif metric == 'mae':
        loss = F.l1_loss(y_pred, y_true, reduction='none')
    else:
        raise ValueError(f"Unsupported metric: {metric}")
        
    # Lấy trung bình lỗi trên tất cả các channel và horizon
    scores = loss.mean(dim=(1, 2))
    return scores.cpu().numpy()
