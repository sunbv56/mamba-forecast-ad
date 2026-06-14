# 🚀 Hướng Dẫn Chạy & Cấu Hình Chương Trình (User & Configuration Guide)

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, chuẩn bị dữ liệu, cấu hình các tham số và chạy huấn luyện (training) cũng như đánh giá (evaluation) hệ thống chẩn đoán lỗi vòng bi dựa trên mô hình lai Mamba-CNN.

---

## 1. 📋 Yêu Cầu Hệ Thống & Cài Đặt (Prerequisites & Installation)

Kiến trúc State Space Model (Mamba) yêu cầu môi trường tính toán có hỗ trợ GPU NVIDIA (CUDA) để đạt hiệu năng tối ưu và biên dịch tăng tốc phần cứng.

### Yêu cầu phần cứng khuyến nghị:
- **GPU**: NVIDIA kiến trúc Ampere trở lên (Compute Capability SM 8.0+, ví dụ: RTX 30xx/40xx, A100, H100).
- **Hệ điều hành**: Linux hoặc Windows (thông qua WSL2 hoặc cài đặt môi trường C++ Build Tools tương thích).

### Các bước cài đặt:

1. **Khởi tạo và kích hoạt môi trường ảo (Virtual Environment)**:
   ```bash
   python -m venv venv
   # Trên Windows:
   .\venv\Scripts\activate
   # Trên Linux/macOS:
   source venv/bin/activate
   ```

2. **Cài đặt thư viện `mamba-ssm`**:
   ```bash
   pip install mamba-ssm --no-build-isolation
   ```

3. **Cài đặt các gói phụ thuộc khác**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. 💾 Chuẩn Bị Dữ Liệu (Dataset Preparation)

Hệ thống hỗ trợ hai cơ chế chuẩn bị dữ liệu: tải thủ công từ Zenodo và tự động đồng bộ từ Hugging Face Hub.

### Cơ chế 1: Đồng bộ tự động từ Hugging Face (Khuyên dùng)
Hệ thống tích hợp cơ chế tự động đồng bộ qua Hugging Face Hub sử dụng thư viện `huggingface_hub` để tải và sửa lỗi các file dữ liệu trực tiếp vào thư mục `data/processed`.
Khi chạy chương trình lần đầu, hệ thống sẽ tự động quét và tải dữ liệu từ repo: `hunglam015/Paderborn_Bearing_Run-to-Failure_Time-Varying`.

### Cơ chế 2: Tải thủ công từ Zenodo
1. Tải file `B02.zip` (hoặc các file vòng bi tương ứng) từ đường dẫn Zenodo: [Bearing Failure Dataset](https://zenodo.org/doi/10.5281/zenodo.10805042).
2. Giải nén và đặt thư mục dữ liệu vào cấu trúc: `data/raw/B02/` (chứa các tệp `.mat` gốc).
3. Chạy các bước tiền xử lý để xuất dữ liệu ra `data/processed/B02/`.

---

## 3. ⚙️ Hướng Dẫn Cấu Hình Mô Hình (Model Configuration)

Các file cấu hình YAML nằm trong thư mục `configs/` được chia làm 3 nhóm chính: `data`, `model`, và `training`. 

### Ví dụ file cấu hình chuẩn cho mô hình lai Mamba-CNN:

```yaml
data:
  raw_dir: "data/raw/B04"
  processed_dir: "data/processed/B04"
  train_datasets: ["data/processed/B01", "data/processed/B03"] # Các dataset dùng để train
  test_datasets: ["data/processed/B01", "data/processed/B03", "data/processed/B04"]
  sampling_rate: 128000
  highpass_freq: 0                  # 0 nghĩa là tắt lọc thông cao theo thiết lập Q1
  label_strategy: 'rms'
  window_stride: 1024
  lookback: 4096                    # Độ dài cửa sổ lịch sử Lx
  horizon: 512                      # Horizon dự báo H (chuẩn hóa ở mốc 512)
  skip_ratio: 0.10                  # Bỏ qua 10% dữ liệu break-in ban đầu
  train_ratio: 0.20                  # Sử dụng 20% dữ liệu healthy đầu tiên để train/calib

model:
  patch_size: 16                    # Kích thước phân mảnh (patch size P)
  patch_stride: 8                   # Bước nhảy phân mảnh (stride S)
  trend_downsample: 64
  cnn_out_channels: 64
  mamba_d_model: 64
  mamba_n_layer: 4 
  mamba_d_state: 16
  mamba_d_conv: 3
  mamba_expand: 3
  bidirectional: false
  decomp_kernel: 25                 # Kích thước cửa sổ trượt trung bình phân rã chuỗi
  auto_scale_baselines: true        # Tự động co giãn tham số để đảm bảo fair parameter parity
  use_decomposition: true           # Bật phân rã chuỗi (Series Decomposition)
  use_stats: true                   # Kích hoạt Physical Stats Head (8 chỉ số vật lý)

training:
  batch_size: 128
  learning_rate: 5e-4
  epochs: 10
  device: "cuda"
```

### Giải Thích Các Tham Số Cốt Lõi:

- **`highpass_freq`**: Đặt thành `0` để tắt lọc thông cao. Ở cấu hình thực nghiệm cuối cùng của bài báo Q1, bộ lọc thông cao được tắt để tránh làm lu mờ hoặc méo mó các tần số rung động thô ban đầu của vòng bi.
- **`lookback` ($L_x = 4096$) & `horizon` ($H = 512$)**: Chuẩn hóa cửa sổ quan sát quá khứ là 4096 bước thời gian và dự báo tương lai là 512 bước để mô hình đạt độ cân bằng tối ưu giữa độ chính xác và tài nguyên tính toán.
- **`train_ratio` ($0.20$) & `skip_ratio` ($0.10$)**:
  - `skip_ratio: 0.10` bỏ qua 10% dữ liệu đầu tiên (giai đoạn rà khớp máy không ổn định).
  - `train_ratio: 0.20` sử dụng chính xác 20% vòng đời lành mạnh đầu tiên của thiết bị để huấn luyện mô hình dự báo và hiệu chuẩn ngưỡng. Thiết lập này tuân thủ nghiêm ngặt giao thức **Kháng rò rỉ dữ liệu (Leakage-Free Validation)**, tuyệt đối không dùng dữ liệu chứa lỗi hoặc thông tin tương lai trong quá trình huấn luyện và chọn ngưỡng.
- **`auto_scale_baselines: true`**: Tự động co giãn số lượng tham số ẩn của các baseline (LSTM, TCN, PatchTST, ModernTCN) tương đồng với Mamba-Hybrid (~200k - 300k tham số) để so sánh hiệu năng công bằng.
- **Tắt bỏ RevIN và Multi-scale Patching**:
  - **Tắt RevIN**: Trong thực nghiệm bài báo Q1, RevIN bị tắt hoàn toàn. RevIN chuẩn hóa cục bộ từng cửa sổ (trừ trung bình, chia độ lệch chuẩn). Khi vòng bi chuyển sang pha suy thoái nặng, biên độ rung vật lý (RMS) thực tế tăng vọt. Việc RevIN chuẩn hóa tức thời vô tình thu nhỏ biên độ lỗi này về mức bình thường giống hệt chuỗi khỏe mạnh, triệt tiêu tín hiệu suy thoái và làm ẩn đi các dấu hiệu lỗi chớm nở. Biên độ tuyệt đối của tín hiệu phải được giữ nguyên để mô hình phát hiện dị thường nhạy bén.
  - **Tắt Multi-scale Patching**: Phân mảnh đa quy mô làm tăng đáng kể độ phức tạp tính toán và nguy cơ quá khớp trên tập dữ liệu trung bình. Thực nghiệm sử dụng phân mảnh đơn quy mô (Simple Patching với `patch_size=16`, `stride=8`) mang lại F1-score cao nhất và tiết kiệm tài nguyên tốt nhất.

---

## 4. 🏋️ Huấn Luyện & Đánh Giá Mô Hình (Model Training & Evaluation)

### A. Huấn luyện (Training)
Sử dụng script `src/training/train.py` để chạy huấn luyện mô hình Mamba-Hybrid hoặc các baselines:
```bash
python src/training/train.py --config configs/mamba_ts.yaml --model Mamba1-Hybrid
```
Các tham số dòng lệnh quan trọng:
- `--config`: Đường dẫn tới file cấu hình YAML.
- `--model`: Tên mô hình cần huấn luyện (`Mamba1-Hybrid`, `LSTM`, `PatchLSTM`, `ModernTCN`, `PatchTST`, hoặc `all` để chạy toàn bộ).
- `--epochs`: Ghi đè số epoch huấn luyện.
- `--batch_size`: Ghi đè kích thước batch.

*Sau khi huấn luyện kết thúc, checkpoint tốt nhất sẽ được lưu tự động tại `results/models/[model_name]_[config_name]_best.pth`.*

### B. Đánh Giá (Evaluation)
Sử dụng script `src/training/eval.py` để chạy kiểm thử mô hình và tính toán các chỉ số phát hiện bất thường:
```bash
python src/training/eval.py --config configs/mamba_ts.yaml --model_type Mamba1-Hybrid --model_path results/models/mamba1_hybrid_best.pth
```
Hoặc đánh giá so sánh đa mô hình (Multi-Model Comparison):
```bash
python src/training/eval.py --config configs/mamba_ts.yaml --models LSTM,PatchLSTM,ModernTCN,PatchTST,Mamba1-Hybrid --models_dir results/models
```
Đầu ra bao gồm:
- Chỉ số dự báo (Forecasting Metrics): MAE, MSE, RMSE.
- Chỉ số phát hiện bất thường (Anomaly Detection Metrics) ứng với các ngưỡng: 3-Sigma, Robust (IQR), và POT (Peak-Over-Threshold).
- Tài nguyên tiêu thụ: VRAM huấn luyện/suy luận và thời gian xử lý thực tế (ms/sample).

### C. Trực Quan Hóa Trên Jupyter Notebook
Phân tích chuyên sâu chu kỳ suy thoái (Run-to-failure lifecycle) và vẽ biểu đồ so sánh:
1. Chạy Jupyter Lab:
   ```bash
   jupyter lab
   ```
2. Mở notebook: `src/notebooks/final-eval-mamba-forecast-ad.ipynb`.
3. Chạy tuần tự các ô lệnh để vẽ đồ thị so sánh hiệu năng 2x2, đường Anomaly Score và so sánh các ngưỡng động.
