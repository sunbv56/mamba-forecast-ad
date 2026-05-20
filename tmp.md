# 📊 Bảng so sánh kết quả đánh giá mô hình (Kaggle Evaluation Results)

Dưới đây là kết quả đánh giá chi tiết được ghi nhận từ phiên chạy trên môi trường Kaggle đối với các mô hình baseline (**LSTM**, **PatchLSTM**, **ModernTCN**, **iTransformer**).

---

## 📈 1. Hiệu năng dự báo chu kỳ sống (Forecasting Metrics)

Bảng dưới đây so sánh sai số dự báo bước tiến (`horizon = 1024`) giữa giá trị thực tế và giá trị dự báo:

| Model | MAE | MSE | RMSE | MAPE (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | 0.358624 | 0.396639 | 0.629793 | 169.0859% | 0.0553 |
| **PatchLSTM** | 0.367844 | 0.409656 | 0.640044 | 104.6360% | 0.0407 |
| **ModernTCN** | 0.402264 | 0.437373 | 0.661341 | 613.4653% | 0.0524 |
| **iTransformer** | 0.369499 | 0.404597 | 0.636079 | 277.2057% | 0.0663 |
| **Mamba-Hybrid** | 0.366829 | 0.412496 | 0.642259 | 139.7167% | 0.1182 |

---

## 🚨 2. Hiệu năng phát hiện bất thường (Anomaly Detection Metrics)

Bảng so sánh hiệu năng phát hiện bất thường thông qua các phương pháp thiết lập ngưỡng động (GMM, 3-Sigma, Robust, POT):

| Model | F1 (3s) | F1 (rb) | F1 (POT) | AUC (POT) | FAR (POT) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | 0.7796 | 0.8139 | **0.8127** | 0.9713 | 0.0084 | 0.0553 |
| **PatchLSTM** | 0.7771 | 0.8100 | 0.8104 | 0.9701 | 0.0082 | **0.0407** |
| **ModernTCN** | 0.7743 | 0.8118 | 0.8115 | 0.9704 | 0.0081 | 0.0524 |
| **iTransformer** | 0.7768 | 0.8118 | **0.8127** | 0.9711 | **0.0080** | 0.0663 |
| **Mamba-Hybrid** | 0.7782 | 0.8127 | 0.8123 | **0.9714** | 0.0082 | 0.1182 |

---

> [!NOTE]
> * **Nhận xét hiệu năng:** Cả 5 mô hình đều cho kết quả cực kỳ đồng đều với chỉ số **F1 (POT) vượt trên 81%** và **AUC (POT) đạt trên 97%**. Trong đó, **Mamba-Hybrid** đạt chỉ số diện tích dưới đường cong ROC cao nhất (**AUC = 0.9714**), thể hiện khả năng phân tách trạng thái hỏng hóc tối ưu.
> * **Độ trễ:** **PatchLSTM** có độ trễ suy luận nhỏ nhất (chỉ **0.0407 ms** cho mỗi mẫu), trong khi **Mamba-Hybrid** với cơ chế quét kép (bidirectional/selective) có độ trễ lớn hơn một chút (**0.1182 ms**) nhưng đi kèm hiệu năng lọc nhiễu tốt.

