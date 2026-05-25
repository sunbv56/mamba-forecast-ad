# 📂 Mô Tả Kiến Trúc Mã Nguồn (Mamba-Forecast-AD Source Code Description)

Tài liệu này mô tả chi tiết cấu trúc thư mục, chức năng của các mô-đun mã nguồn và luồng dữ liệu của hệ thống phát hiện bất thường dựa trên dự báo chuỗi thời gian.

---

## 1. 📂 Cấu Trúc Tổng Quan Thư Mục Mã Nguồn

```text
src/
├── data/                  # Tiền xử lý và quản lý tập dữ liệu
│   └── dataset.py         # Lớp tải dữ liệu BearingDataset và MultiBearingDataset
├── models/                # Định nghĩa các kiến trúc mô hình học sâu
│   ├── mamba/             # Mô hình đề xuất lai Mamba-CNN và biến thể MambaTS
│   │   ├── hybrid_mamba.py       # Mô hình lai HybridMambaCNN (Đóng góp cốt lõi)
│   │   └── mamba_ts_official.py  # Bản cài đặt baseline MambaTS chính thức
│   └── baselines/         # Các mô hình đối chứng (LSTM, ModernTCN, PatchTST...)
├── training/              # Vòng lặp huấn luyện, tối ưu và dừng sớm
│   ├── train.py           # Script chạy huấn luyện và tìm ngân sách tham số đối chứng
│   └── eval.py            # Script chạy đánh giá đa mô hình trên tập kiểm thử
├── evaluation/            # Các mô-đun tính toán chỉ số phát hiện bất thường
│   ├── anomaly_scorer.py  # Tính điểm bất thường (Anomaly Score) từ sai số dự báo
│   └── metrics.py         # Cài đặt các thuật toán xác định ngưỡng (POT, Robust, GMM...)
└── utils/                 # Các thư viện bổ trợ trực quan hóa và vẽ đồ thị
```

---

## 2. 🧩 Chi Tiết Các Mô-đun và Lớp Cốt Lõi

### A. Mô-đun Tiền Xử Lý Dữ Liệu (`src/data/dataset.py`)

Mô-đun chịu trách nhiệm đọc dữ liệu thô, lọc nhiễu, chuẩn hóa và đóng gói thành các cửa sổ trượt.

- **`BearingDataset`**: Quản lý dữ liệu của một vòng bi đơn lẻ.
  - *Lọc thông dải*: Áp dụng bộ lọc Butterworth thông cao (`highpass_freq: 2000` Hz) để triệt tiêu tần số thấp của động cơ nền.
  - *Tính toán đặc trưng thống kê (Stats Head)*: Trích xuất 8 thông số vật lý của cửa sổ lịch sử: RMS (Root Mean Square), Kurtosis, Crest Factor, Shape Factor, Impulse Factor, Margin Factor, Peak-to-Peak, và Variance. Các đặc trưng này đóng vai trò là tri thức dẫn hướng vật lý (Physics-Informed features).
  - *Phân chia cửa sổ*: Trượt cửa sổ lookback (đầu vào) và horizon (nhãn dự báo) để chuẩn bị cho bài toán tự giám sát.
- **`MultiBearingDataset`**: Lớp bao bọc (Wrapper) kết hợp nhiều đối tượng `BearingDataset` để phục vụ huấn luyện mô hình trên đa thiết bị (Generalization).
  - *Ngăn ngừa rò rỉ dữ liệu*: Thống kê chuẩn hóa (mean, std) được tính toán **chỉ trên tập huấn luyện khỏe mạnh** và truyền sang tập kiểm thử thông qua tham số `oc_stats`. Điều này đảm bảo dữ liệu lỗi của tập kiểm thử không gây ảnh hưởng đến phân phối chuẩn hóa của tập huấn luyện.

---

### B. Mô-đun Kiến Trúc Mô Hình đề xuất (`src/models/mamba/hybrid_mamba.py`)

- **`HybridMambaCNN`**: Trọng tâm nghiên cứu của bài báo.
  1. **Series Decomposition (Phân rã chuỗi)**: Tín hiệu đầu vào được phân rã thành thành phần xu thế (`Trend`) thông qua bộ lọc trung bình trượt và thành phần chu kỳ (`Seasonal`).
  2. **Patching & CNN Encoder**: Thành phần Seasonal được cắt thành các đoạn nhỏ (Patches) và đưa qua các lớp tích chập 1D để trích xuất đặc trưng không gian cục bộ (Local Features).
  3. **Mamba Encoder Block**: Chuỗi đặc trưng sau tích chập được đưa qua các khối Mamba (State Space Model) để mô hình hóa mối quan hệ phụ thuộc xa (Long-range dependencies) với độ phức tạp tuyến tính $O(N)$.
  4. **Stats Projection Head & Fusion Head**: Nhánh Stats Head trích xuất đặc trưng vật lý trực tiếp từ chuỗi thô, đi qua một mạng tuyến tính chiếu (Projection Head), sau đó được ghép nối (Concatenate) với đầu ra của Mamba Encoder trước khi đưa vào lớp dự báo cuối cùng để đưa ra dự báo tín hiệu tương lai.

---

### C. Mô-đun Huấn Luyện & Dừng Sớm (`src/training/train.py`)

- **`train_one_model`**: Vòng lặp huấn luyện tối ưu hóa tham số mô hình.
  - *Hàm mất mát*: Sử dụng `HuberLoss` để nâng cao tính bền vững trước nhiễu gai.
  - *Tối ưu hóa tốc độ học*: Sử dụng `CosineAnnealingLR` kết hợp dừng sớm `EarlyStopping` dựa trên sai số của tập Validation.
- **Tự động co giãn tham số (`auto_scale_baselines`)**:
  - Chứa các hàm tìm kiếm kích thước mạng phù hợp: `find_closest_lstm`, `find_closest_patch_lstm`, `find_closest_modern_tcn`, `find_closest_patchtst`.
  - Tính toán tổng tham số của mô hình Mamba lai, sau đó tự động điều chỉnh tham số chiều ẩn của các baselines tương ứng để đảm bảo tính công bằng thực nghiệm.

---

## 3. 🔄 Luồng Dữ Liệu Hoạt Động (Data Pipeline Flow)

Luồng xử lý từ dữ liệu thô đến việc ra quyết định cảnh báo lỗi được mô tả theo sơ đồ dưới đây:

```mermaid
graph TD
    A[Raw Vibration Signal 128kHz] --> B[Butterworth High-pass Filter 2kHz]
    B --> C[Temporal Sliding Windowing]
    C --> D[Lookback Window: L=4096]
    C --> E[Horizon Window: H=1024]
    
    D --> F[1. Physical Stats Extraction: Stats Head]
    D --> G[2. Series Decomposition: Trend & Seasonal]
    
    G --> H[CNN Patching Encoder]
    H --> I[Mamba SSM Layers]
    
    I --> J[Fusion Head]
    F --> J
    
    J --> K[Forecasted Signal Output]
    K & E --> L[Anomaly Scorer: Calculate MSE]
    
    L --> M[Anomaly Score Flow]
    M --> N[Per-Bearing Threshold Calibration: POT / Robust]
    N --> O[Dynamic Alarm Decision: Normal / Anomaly]
```
