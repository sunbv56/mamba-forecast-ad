# Cấu trúc thư mục dự án: Mamba-SFT (Sinh viên AI 2)

Dựa trên nhiệm vụ chính là **Code beta, experiment automation và visualization**, dưới đây là cấu trúc thư mục tối ưu để phát triển dự án:

```text
Mamba-SFT/
├── data/                             # Nơi chứa dữ liệu
│   ├── raw/                          # Chứa các file gốc của dataset
│   ├── processed/                    # Dữ liệu sau Windowing & Normalize
│   └── splits/                       # Lưu các index chia train/val/test
│
├── configs/                          # Các file cấu hình YAML/JSON
│   ├── default.yaml                  # Cấu hình mặc định
│   └── experiment_configs/           # Cấu hình cho việc auto-run (lookback, horizon, v.v.)
│
├── src/                              # Mã nguồn chính của dự án (Code beta)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # Load dataset, Build forecasting pairs (X quá khứ, Y tương lai)
│   │   └── pipeline.py               # Thực hiện Pipeline: Windowing, Normalize
│   │
│   ├── models/                       # Các kiến trúc Baseline và kiến trúc chính
│   │   ├── __init__.py
│   │   ├── baselines/                # Thư mục chứa các baseline
│   │   │   ├── lstm.py               # Baseline LSTM
│   │   │   ├── tcn.py                # Baseline TCN
│   │   │   └── transformer_small.py  # Baseline Transformer nhỏ
│   │   └── mamba/                    # Mô hình Mamba (nếu triển khai kịp)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                # Vòng lặp Train model, Evaluate anomaly score
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── anomaly_scorer.py         # Tính toán Anomaly score từ sai số dự báo
│   │   └── metrics.py                # Tính Detection delay, False alarm rate, F1, AUC
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py          # Draw chart (Confusion matrix, Anomaly score over TTF%, Threshold chart)
│       └── logger.py                 # Lưu CSV log các experiment
│
├── scripts/                          # Script tự động chạy nhiều cấu hình
│   ├── run_pipeline.sh               # Chạy toàn bộ pipeline chuẩn bị dữ liệu
│   └── auto_run_experiments.sh       # Auto-run quét qua lookback N, horizon K, patch size, LR, batch size, hidden dim
│
├── results/                          # Nơi xuất kết quả
│   ├── logs/                         # CSV log của các experiment
│   ├── metrics/                      # Confusion matrix, bảng so sánh baseline
│   └── plots/                        # Biểu đồ Anomaly score over TTF%, Threshold chart
│
├── main.py                           # Entry point chính để chạy dự án
└── requirements.txt                  # Danh sách thư viện cần thiết
```

## Giải thích chức năng theo Nhiệm vụ:

1. **Pipeline Dữ liệu (`src/data/`)**: Đảm nhiệm toàn bộ quy trình: Load dataset, Windowing, Normalize và Build forecasting pairs (đầu vào $X$ quá khứ, đầu ra $Y$ tương lai).
2. **Triển khai Mô hình (`src/models/`)**: Dựng source code beta cho các baseline (LSTM, TCN, Transformer nhỏ) và Mamba.
3. **Training & Đánh giá (`src/training/` & `src/evaluation/`)**: Huấn luyện mô hình và đánh giá chuyên sâu các chỉ số (Anomaly score, Detection delay, False alarm rate).
4. **Tự động hóa (`scripts/auto_run_experiments.sh`)**: Tự động hóa quá trình huấn luyện và đánh giá trên nhiều cấu hình (lookback N, horizon K, patch size, learning rate, batch size, hidden dimension).
5. **Trực quan hóa và Kết quả (`src/utils/visualization.py` & `results/`)**: Chịu trách nhiệm lưu CSV log, vẽ Confusion matrix, biểu đồ Anomaly Score over TTF%, và Threshold chart phục vụ cho việc phân tích.
