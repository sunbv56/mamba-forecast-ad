import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

def calculate_threshold_3sigma(healthy_scores):
    """
    Tính ngưỡng dự báo lỗi dựa trên nguyên tắc 3-sigma (Mean + 3*Std) của tập Healthy.
    """
    mu = np.mean(healthy_scores)
    sigma = np.std(healthy_scores)
    threshold = mu + 3 * sigma
    return threshold

def calculate_metrics(anomaly_scores, true_labels, threshold):
    """
    Tính toán các chỉ số: Detection Delay, FAR (False Alarm Rate), F1, AUC
    anomaly_scores: mảng 1D các điểm số bất thường theo thời gian.
    true_labels: mảng 1D nhãn thực (0 = Healthy, 1 = Fault/Degrading).
    threshold: ngưỡng cảnh báo tính từ hàm calculate_threshold_3sigma.
    """
    preds = (anomaly_scores > threshold).astype(int)
    
    f1 = f1_score(true_labels, preds)
    try:
        auc = roc_auc_score(true_labels, anomaly_scores)
    except ValueError:
        auc = 0.5
        
    cm = confusion_matrix(true_labels, preds, labels=[0, 1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn + 1e-8)
    else:
        far = 0.0
        tn, fp, fn, tp = 0, 0, 0, 0
        
    # Tính Detection Delay: Khoảng thời gian từ lúc có lỗi thật sự tới khi mô hình phát hiện được lỗi đầu tiên
    fault_idx = np.argmax(true_labels == 1) if 1 in true_labels else -1
    
    pred_faults = np.where((preds == 1) & (true_labels == 1))[0]
    pred_fault_idx = pred_faults[0] if len(pred_faults) > 0 else -1
    
    if fault_idx != -1 and pred_fault_idx != -1:
        delay = max(0, int(pred_fault_idx - fault_idx))
    else:
        delay = -1 # Không bắt được hoặc dataset không có lỗi
        
    return {
        "F1": f1,
        "AUC": auc,
        "FAR": far,
        "Delay": delay,
        "Threshold": threshold,
        "CM": cm.tolist()
    }
