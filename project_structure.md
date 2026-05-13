# Cấu trúc thư mục dự án: Mamba-Forecast-AD

Dưới đây là cấu trúc thư mục thực tế của dự án, tập trung vào việc dự báo và phát hiện bất thường sử dụng kiến trúc MambaTS:

```text
Mamba-Forecast-AD/
├── data/                             # Nơi chứa dữ liệu
│   ├── raw/                          # Chứa các file .mat gốc của dataset B02
│   └── processed/                    # Dữ liệu Tensor .pt và Operating Conditions
│
├── configs/                          # Các file cấu hình YAML
│   └── mamba_ts.yaml                 # Cấu hình chi tiết cho model và pipeline
│
├── src/                              # Mã nguồn chính của dự án
│   ├── data/
│   │   ├── dataset.py                # B02Dataset: Xử lý Windowing, Normalize, RMS Labeling, Filter
│   │   └── pipeline.py               # Preprocess: Chuyển đổi .mat sang .pt
│   │
│   ├── models/                       # Các kiến trúc mô hình
│   │   └── mamba/                    # MambaTS và các biến thể
│   │       ├── mamba_ts.py           # MambaTS (Official style): Patching, VAS, TMB (Multi-variate correlation)
│   │       ├── mamba_ts_official.py  # Bản triển khai tham chiếu từ paper gốc
│   │       ├── hybrid_mamba.py       # HybridMamba (CI-Mamba++): Decomposition, Multi-scale (Primary/Recommended)
│   │       └── layers.py             # Các lớp bổ trợ (RevIN, Decomposition, Patching)
│   │
│   ├── training/                     # Huấn luyện và Đánh giá (Điểm bắt đầu chính)
│   │   ├── train.py                  # Script huấn luyện chính
│   │   ├── trainer.py                # Lớp Trainer đóng gói logic huấn luyện
│   │   ├── eval.py                   # Script đánh giá và tính anomaly score
│   │   └── losses.py                 # Các hàm loss (MSE, MAE, v.v.)
│   │
│   ├── evaluation/                   # Đánh giá chuyên sâu
│   │   ├── anomaly_scorer.py         # Tính toán Anomaly score (MSE, Log-MSE)
│   │   ├── thresholding.py           # Các phương pháp tính ngưỡng (GMM, POT)
│   │   ├── metrics.py                # Chỉ số: F1, AUC, FAR, Detection Delay
│   │   ├── visualize_trend.py        # Vẽ biểu đồ xu hướng suy giảm toàn cục
│   │   └── visualize_file.py         # Vẽ chi tiết từng file tín hiệu
│   │
│   └── utils/                        # Công cụ hỗ trợ
│       └── logger.py                 # Ghi log experiment
│
├── scripts/                          # Script tự động hóa
│   ├── precompute_rms.py             # Tính toán trước RMS để tăng tốc loading
│   ├── run_pipeline.sh               # Chạy toàn bộ luồng xử lý
│   └── auto_run_experiments.sh       # Quét hyperparameter tự động
│
├── results/                          # Nơi lưu kết quả (Models, Plots, Logs)
├── main.py                           # (Tạm thời không sử dụng)
├── test.py                           # (Tạm thời không sử dụng)
└── requirements.txt                  # Danh sách thư viện
```

## Giải thích chức năng chính:

1.  **Dữ liệu (`src/data/`)**: `B02Dataset` thực hiện chia dữ liệu **10/50/40**, lọc thông cao, tính RMS vật lý để gán nhãn ground truth, và thực hiện RevIN (Window Normalization).
2.  **Mô hình (`src/models/mamba/`)**: Triển khai `MambaTS` với cơ chế **Patching**, **Variable-Aware Scanning (VAS)** và các khối **Temporal Mamba**.
3.  **Huấn luyện (`src/training/`)**: Tập trung vào nhiệm vụ dự báo chuỗi thời gian (Forecasting) để học quy luật của dữ liệu lành mạnh.
4.  **Đánh giá (`src/evaluation/`)**: Chuyển đổi sai số dự báo thành điểm bất thường, tính toán ngưỡng động bằng **POT** hoặc **GMM**, và đo lường các chỉ số chẩn đoán lỗi công nghiệp.
5.  **Trực quan hóa**: Cung cấp các công cụ so sánh điểm bất thường của mô hình với giá trị RMS vật lý để kiểm chứng tính đúng đắn về mặt cơ học.
