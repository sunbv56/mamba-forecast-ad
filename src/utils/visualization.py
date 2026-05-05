import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def plot_anomaly_score_over_ttf(anomaly_scores, threshold, save_path=None):
    """
    Vẽ biểu đồ Anomaly Score theo phần trăm Time-to-Failure (TTF).
    """
    n = len(anomaly_scores)
    ttf_percent = np.linspace(0, 100, n)
    
    plt.figure(figsize=(10, 5))
    plt.plot(ttf_percent, anomaly_scores, label="Anomaly Score", color='#1f77b4', alpha=0.8, linewidth=1.5)
    plt.axhline(y=threshold, color='#d62728', linestyle='--', linewidth=2, label=f"Threshold ({threshold:.4f})")
    
    plt.title("Anomaly Score over Time-to-Failure (%)", fontsize=14)
    plt.xlabel("TTF (%) - 100% is Failure", fontsize=12)
    plt.ylabel("Anomaly Score (MSE)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix_custom(cm, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Fault'], yticklabels=['Healthy', 'Fault'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
