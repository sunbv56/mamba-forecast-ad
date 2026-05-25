# 🚀 Hướng Dẫn Chạy Chương Trình (Mamba-Forecast-AD Running Guide)

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, chuẩn bị dữ liệu, chạy huấn luyện (training) và chạy đánh giá (evaluation) cho hệ thống phát hiện bất thường dựa trên mô hình lai Mamba-CNN.

---

## 1. 📋 Yêu Cầu Hệ Thống & Cài Đặt (Prerequisites & Installation)

Hệ thống Mamba yêu cầu môi trường tính toán có hỗ trợ GPU NVIDIA (CUDA) để đạt hiệu năng tối ưu và tránh lỗi biên dịch thư viện `mamba-ssm`.

### Yêu cầu phần cứng khuyến nghị:
- GPU NVIDIA kiến trúc Ampere trở lên (Compute Capability SM 8.0+, ví dụ: RTX 30xx/40xx, A100, H100).
- Hệ điều hành: Linux hoặc Windows (thông qua WSL2 hoặc cài đặt môi trường C++ thích hợp).

### Các bước cài đặt:

1. **Khởi tạo và kích hoạt môi trường ảo (Virtual Environment)**:
   ```bash
   python -m venv venv
   # Trên Windows:
   .\venv\Scripts\activate
   # Trên Linux/macOS:
   source venv/bin/activate
   ```

2. **Cài đặt thư viện `mamba-ssm` (Không yêu cầu biên dịch lại)**:
   ```bash
   pip install mamba-ssm --no-build-isolation
   ```

3. **Cài đặt các gói phụ thuộc khác**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 2. 💾 Chuẩn Bị Dữ Liệu (Dataset Preparation)

Dự án hỗ trợ hai cơ chế chuẩn bị dữ liệu: tải thủ công và tự động đồng bộ từ Hugging Face Hub.

### Cơ chế 1: Đồng bộ tự động từ Hugging Face (Khuyên dùng)
Hệ thống tích hợp hàm `sync_dataset` sử dụng thư viện `huggingface_hub` để tự động quét, tải và sửa lỗi các file dữ liệu tại thư mục `data/processed`.
Nếu bạn chạy chương trình lần đầu, hệ thống sẽ tự động kéo dữ liệu từ repo dataset: `hunglam015/Paderborn_Bearing_Run-to-Failure_Time-Varying` phân bộ `processed`.

### Cơ chế 2: Tải thủ công từ Zenodo
1. Tải file `B02.zip` từ đường dẫn Zenodo: [Bearing Failure Dataset (B02)](https://zenodo.org/doi/10.5281/zenodo.10805042).
2. Giải nén và đặt thư mục dữ liệu vào cấu trúc: `data/raw/B02/`.
3. Chạy các bước tiền xử lý để xuất dữ liệu ra `data/processed/B02/`.

---

## 3. 🏋️ Huấn Luyện Mô Hình (Model Training)

Sử dụng script `src/training/train.py` để thực hiện huấn luyện mô hình Mamba và các baselines.

### Cú pháp cơ bản:
```bash
python src/training/train.py --config configs/nano.yaml --model Mamba1-Hybrid
```

### Các tham số dòng lệnh quan trọng:
- `--config`: Đường dẫn tới file cấu hình YAML (ví dụ: `configs/nano.yaml`, `configs/snano.yaml`, v.v.).
- `--model`: Tên mô hình cần huấn luyện (`Mamba1-Hybrid`, `LSTM`, `PatchLSTM`, `ModernTCN`, `PatchTST` hoặc `all` để huấn luyện tất cả tuần tự).
- `--epochs`: Ghi đè số lượng epoch huấn luyện từ cấu hình YAML (ví dụ: `--epochs 20`).
- `--batch_size`: Ghi đè kích thước batch (ví dụ: `--batch_size 128`).
- `--file_subset_ratio`: Tỷ lệ lấy mẫu tệp dữ liệu để tăng tốc độ huấn luyện mà vẫn giữ tính liên tục thời gian (Ví dụ: `--file_subset_ratio 10` sẽ lấy mẫu cứ mỗi 10 file dữ liệu khỏe mạnh).

*Sau khi kết thúc huấn luyện, checkpoint tốt nhất sẽ được lưu tự động tại `results/models/[model_name]_[config_name]_best.pth` nhờ cơ chế Early Stopping.*

---

## 4. 📊 Đánh Giá Mô Hình (Model Evaluation)

Sử dụng script `src/training/eval.py` để chạy kiểm thử mô hình trên các vòng bi khác nhau và tính toán các chỉ số phát hiện bất thường.

### Cách 1: Đánh giá một mô hình đơn lẻ (Single Model Evaluation)
Đánh giá hiệu năng của một file checkpoint cụ thể:
```bash
python src/training/eval.py --config configs/nano.yaml --model_type Mamba1-Hybrid --model_path results/models/mamba1_hybrid_nano_best.pth
```

### Cách 2: Đánh giá so sánh đa mô hình (Multi-Model Comparison)
Tự động quét thư mục lưu trữ checkpoints để đánh giá so sánh hiệu năng dự báo và phát hiện bất thường của tất cả các mô hình:
```bash
python src/training/eval.py --config configs/nano.yaml --models LSTM,PatchLSTM,ModernTCN,PatchTST,Mamba1-Hybrid --models_dir results/models
```

**Đầu ra kết quả bao gồm:**
- Bảng so sánh sai số dự báo (Forecasting Metrics): MAE, MSE, RMSE, MAPE.
- Bảng so sánh phát hiện bất thường (Anomaly Detection Metrics) ứng với các ngưỡng: 3-Sigma, Robust, POT (Peak-Over-Threshold), Self-Learn (GMM), và Optimal.
- Tốc độ xử lý trung bình (ms/sample) của từng mô hình.

---

## 5. 📓 Trực Quan Hóa Trên Jupyter Notebook

Để phân tích chuyên sâu, vẽ các biểu đồ phân phối điểm lỗi và biểu diễn chu kỳ suy thoái (Run-to-failure lifecycle) của vòng bi:

1. Khởi chạy Jupyter Lab / Notebook:
   ```bash
   jupyter lab
   ```
2. Mở file `src/notebooks/final-eval-mamba-forecast-ad.ipynb`.
3. Chạy tuần tự các ô lệnh để vẽ đồ thị so sánh hiệu năng 2x2 giữa Mamba và các baselines, luồng điểm bất thường thời gian thực (Anomaly Score Flow) và so sánh ngưỡng động.
