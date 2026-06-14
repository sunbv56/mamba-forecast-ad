# 🛠️ Tài Liệu Kỹ Thuật Chi Tiết (Technical Reference & Mathematical Flow)

Tài liệu này trình bày chi tiết về cấu trúc thư mục, chức năng của từng file mã nguồn, quy trình xử lý dữ liệu và cơ sở toán học của kiến trúc **Mamba-Hybrid (CI-Mamba++ with Decomposition & Stats)**.

---

## 1. 📂 Cấu Trúc Thư Mục Thực Tế (Project Structure)

```text
Mamba-Forecast-AD/
├── data/                             # Nơi chứa dữ liệu
│   ├── raw/                          # Các file .mat gốc của bộ dữ liệu Paderborn (UPB)
│   └── processed/                    # Dữ liệu Tensor .pt và Operating Conditions sau tiền xử lý
│
├── configs/                          # Các cấu hình YAML
│   └── mamba_ts.yaml                 # Cấu hình chi tiết cho model và pipeline thực nghiệm
│
├── docs/                             # Tài liệu hướng dẫn và tham chiếu kỹ thuật
│   ├── user_guide.md                 # Hướng dẫn cài đặt, cấu hình và chạy chương trình
│   ├── technical_reference.md        # [Tài liệu này] Chi tiết mã nguồn và công thức toán học
│   └── review_guide_Q1.md            # Hướng dẫn và checklist tự bình duyệt chuẩn Q1
│
├── src/                              # Mã nguồn chính của dự án
│   ├── data/
│   │   ├── dataset.py                # B02Dataset: Xử lý Windowing, RMS Labeling, Filter
│   │   └── pipeline.py               # Preprocess: Đồng bộ và chuyển đổi .mat sang .pt
│   │
│   ├── models/                       # Các kiến trúc mô hình
│   │   └── mamba/
│   │       ├── hybrid_mamba.py       # HybridMamba (CI-Mamba++): Decomposition, Stats Head (Khuyên dùng)
│   │       ├── mamba_ts.py           # MambaTS (Official style): Patching, VAS, TMB
│   │       ├── mamba_ts_official.py  # Bản triển khai tham chiếu từ paper gốc
│   │       └── layers.py             # Các lớp bổ trợ (Decomposition, Patching, StatsHead)
│   │
│   ├── training/                     # Huấn luyện và Đánh giá chính
│   │   ├── train.py                  # Script huấn luyện chính (hỗ trợ auto-scale baselines)
│   │   ├── trainer.py                # Lớp Trainer đóng gói logic huấn luyện và early stopping
│   │   ├── eval.py                   # Script đánh giá, tính anomaly score và so sánh đa mô hình
│   │   └── losses.py                 # Hàm loss tùy chỉnh (MSE, MAE, HuberLoss)
│   │
│   ├── evaluation/                   # Đánh giá chuyên sâu
│   │   ├── anomaly_scorer.py         # Tính toán Anomaly score (MSE, Log-MSE)
│   │   ├── thresholding.py           # Các phương pháp tính ngưỡng (3-Sigma, Robust MAD, POT GPD, GMM)
│   │   ├── metrics.py                # Chỉ số chẩn đoán: F1, FAR, Detection Delay
│   │   ├── visualize_trend.py        # Vẽ biểu đồ xu hướng suy giảm toàn vòng đời
│   │   └── visualize_file.py         # Vẽ chi tiết tín hiệu của từng file
│   │
│   └── utils/                        # Công cụ hỗ trợ
│       └── logger.py                 # Ghi log thực nghiệm
│
├── results/                          # Nơi lưu kết quả (Checkpoints mô hình, Plots trực quan, Logs)
├── requirements.txt                  # Danh sách thư viện phụ thuộc
└── README.md                         # Tài liệu trang chủ chính của dự án
```

---

## 2. 📝 Mô Tả Chức Năng Mã Nguồn (Source Code Description)

### A. Module Dữ Liệu (`src/data/`)
- **`dataset.py`**: Định nghĩa lớp `B02Dataset`. Thực hiện chia cửa sổ trượt (sliding windowing), chuẩn hóa, lọc thông cao và tính toán giá trị RMS vật lý thời gian thực để gán nhãn Ground Truth bất thường của vòng bi.
- **`pipeline.py`**: Quản lý quy trình xử lý dữ liệu thô. Chuyển đổi các file MATLAB `.mat` của tập dữ liệu Paderborn thành Tensor PyTorch `.pt` để tăng tốc độ đọc từ ổ cứng, tích hợp hàm `sync_dataset` để tự động kéo dữ liệu bị thiếu từ Hugging Face Hub.

### B. Module Kiến Trúc Mô Hình (`src/models/mamba/`)
- **`hybrid_mamba.py`**: Triển khai lớp `HybridMamba` (được khuyến nghị sử dụng chính). Tích hợp cơ chế phân rã chuỗi thời gian (Series Decomposition), xử lý phân mảnh đơn quy mô (Simple Patching), State Space Model (Mamba) độc lập kênh (Channel-Independent), và đầu thông tin vật lý 8 chỉ số (Physical Stats Head) để đưa ra dự báo tín hiệu tương lai.
- **`layers.py`**: Chứa các khối thành phần tái sử dụng bao gồm: `SeriesDecomposition` (sử dụng Moving Average pooling), `SimplePatchEmbedding`, và `PhysicalStatsHead`.

### C. Module Huấn Luyện (`src/training/`)
- **`train.py`**: Script huấn luyện chính. Hỗ trợ cơ chế tự động co giãn tham số baselines (`auto_scale_baselines`), giúp tự động co giãn số chiều ẩn của các baseline (LSTM, TCN, PatchTST, ModernTCN) để có cùng parameter budget với Mamba-Hybrid, đảm bảo so sánh công bằng.
- **`trainer.py`**: Quản lý vòng lặp huấn luyện, tính toán loss trên tập Validation, và kích hoạt Early Stopping khi loss không còn cải thiện để tránh overfitting.
- **`eval.py`**: Thực hiện suy luận (inference), tính toán sai số dự báo làm Anomaly Score, hiệu chuẩn các ngưỡng phát hiện lỗi, và lưu kết quả so sánh hiệu năng.

### D. Module Đánh Giá (`src/evaluation/`)
- **`anomaly_scorer.py`**: Tính toán Anomaly Score dựa trên MSE sai số dự báo giữa tín hiệu thực tế và tín hiệu mô hình dự báo.
- **`thresholding.py`**: Triển khai các thuật toán hiệu chuẩn ngưỡng phát hiện lỗi, bao gồm cả thuật toán Peak Over Threshold (POT) dựa trên Extreme Value Theory (EVT).
- **`metrics.py`**: Đo lường các chỉ số hiệu năng công nghiệp bao gồm F1-Score, Tỷ lệ báo động giả (FAR), và Độ trễ phát hiện (Detection Delay).

---

## 3. 📐 Cơ Sở Toán Học & Luồng Xử Lý (Mathematical Flow)

### 3.1. Quy trình Xử lý Dữ liệu (Data Processing Logic)

#### Cặp dữ liệu Dự báo (Forecasting Pairs)
Với tín hiệu rung động thô đa biến $S \in \mathbb{R}^{C \times L_{total}}$ (với $C=2$ kênh cảm biến radial/axial):
- **Cửa sổ đầu vào (Lookback Window)**: $X = S[:, t:t+L_x] \in \mathbb{R}^{C \times L_x}$
- **Cửa sổ đích cần dự báo (Horizon Window)**: $Y = S[:, t+L_x:t+L_x+H] \in \mathbb{R}^{C \times H}$
Trong đó: $L_x = 4096$ là chiều dài quá khứ, $H = 512$ là chiều dài tương lai cần dự báo.

#### Phân chia Dữ liệu Kháng rò rỉ (Leakage-Free Temporal Splitting)
Để tránh rò rỉ thông tin lỗi vào quá trình huấn luyện và hiệu chuẩn ngưỡng:
1. **Pha Huấn luyện & Hiệu chuẩn Ngưỡng (20% Vòng đời đầu)**: Chỉ chứa các tín hiệu ở trạng thái hoàn toàn lành mạnh (Healthy State). Mô hình được huấn luyện để học quy luật dự báo tín hiệu lành mạnh. Ngưỡng báo động lỗi cũng được hiệu chuẩn chỉ dựa trên phân phối sai số của tập này.
2. **Pha Kiểm thử (80% Vòng đời còn lại)**: Chứa cả dữ liệu khỏe mạnh còn lại và toàn bộ tiến trình suy thoái cho tới khi hỏng hoàn toàn (Run-to-failure), dùng để đánh giá độ trễ phát hiện lỗi và tỷ lệ báo động giả thực tế.

---

### 3.2. Công thức Kiến trúc Mamba-Hybrid (Proposed Architecture Equations)

Kiến trúc mô hình được xây dựng theo sơ đồ toán học tuần tự sau:

#### Bước 1: Phân tách Chuỗi (Series Decomposition)
Đầu vào $X \in \mathbb{R}^{B \times C \times L_x}$ được tách thành hai phần: xu hướng tần số thấp (Trend) và dao động tần số cao (Seasonal) bằng bộ lọc trung bình trượt (Moving Average):

$$
X_{\text{trend}} = \text{AvgPool1d}(X, \text{kernel\_size}=25)
$$

$$
X_{\text{seasonal}} = X - X_{\text{trend}}
$$

#### Bước 2: Phân mảnh Đơn Quy mô (Simple Patch Embedding)
Thành phần Seasonal $X_{\text{seasonal}}$ được chia thành các mảnh (patches) kích thước $P=16$, bước nhảy $S=8$:

$$
N = \left\lfloor \frac{L_x - P}{S} \right\rfloor + 1
$$

Mỗi mảnh được chiếu tuyến tính lên không gian ẩn chiều $D$ (với `d_model = 64`):

$$
s_{\text{seasonal}} \in \mathbb{R}^{B \times C \times N \times D}
$$

#### Bước 3: Độc lập Kênh & Khối Mamba Encoder (Channel-Independent Mamba)
Chiều kênh $C$ được đưa vào batch để chia sẻ trọng số và triệt tiêu nhiễu chéo:

$$
s_{\text{folded}} \in \mathbb{R}^{(B \cdot C) \times N \times D}
$$

Đưa qua mạng Mamba Encoder gồm $N_{layer}=4$ khối Selective State Space Model để học tương quan thời gian dài:

$$
h(t) = \mathbf{A}(t) h(t-1) + \mathbf{B}(t) s_{\text{folded}}(t)
$$

$$
\hat{s}(t) = \mathbf{C}(t) h(t) + \mathbf{D} s_{\text{folded}}(t)
$$

$$
s_{\text{hidden}} \in \mathbb{R}^{(B \cdot C) \times N \times D}
$$

#### Bước 4: Hợp nhất Đầu Thông số Vật lý (Physical Stats Head Fusion)
Để bổ trợ tri thức cơ học và tăng tính tường minh giải thích, $8$ đặc trưng thống kê miền thời gian được trích xuất trực tiếp từ cửa sổ Lookback thô của nhánh tương ứng:

$$
stats = [\text{Mean}, \text{Std}, \text{RMS}, \text{Peak-to-Peak}, \text{Skewness}, \text{Kurtosis}, \text{Crest Factor}, \text{Shape Factor}]
$$

Trong đó, **Kurtosis (Độ nhọn)** đóng vai trò quan trọng phát hiện xung va đập chớm lỗi:

$$
\text{Kurtosis} = \frac{\frac{1}{L_x}\sum_{i=1}^{L_x} (x_i - \mu)^4}{\sigma^4}
$$

Đặc trưng vật lý được reshape thành dạng folded: $stats_{\text{folded}} \in \mathbb{R}^{(B \cdot C) \times 8}$. Vector ẩn từ Mamba được làm phẳng và concat trực tiếp với đặc trưng vật lý được chiếu tuyến tính:

$$
s_{\text{flat}} = \text{Flatten}(s_{\text{hidden}}) \in \mathbb{R}^{(B \cdot C) \times (N \cdot D)}
$$

$$
s_{\text{fused}} = \text{Concat}\Big(s_{\text{flat}}, \text{LinearProjection}(stats_{\text{folded}})\Big)
$$

$$
y_{\text{seasonal\_folded}} = \text{LinearProjection}(s_{\text{fused}} \to H)
$$

Sau đó, tiến hành khôi phục lại chiều kênh (Unfolding) để thu được dự báo nhánh Seasonal:

$$
y_{\text{seasonal}} \in \mathbb{R}^{B \times C \times H}
$$

#### Bước 5: Trộn Thích ứng Học được (Learnable Mixing Layer)
Nhánh Trend được dự báo riêng bằng lớp Linear siêu nhẹ:

$$
y_{\text{trend}} = \text{LinearProjection}(\text{AvgPool1d}(X_{\text{trend}})) \in \mathbb{R}^{B \times C \times H}
$$

Kết quả cuối cùng $y_{\text{forecast}}$ được trộn thích ứng theo từng kênh $c$ bằng trọng số $\alpha_c$ học được qua hàm Sigmoid:

$$
\alpha_{c} = \text{Sigmoid}(w_{c}) \quad (w_c \in \mathbb{R} \text{ là tham số học được})
$$

$$
y_{\text{forecast}, c} = \alpha_{c} \cdot y_{\text{seasonal}, c} + (1 - \alpha_{c}) \cdot y_{\text{trend}, c}
$$

---

### 3.3. Quy trình Phát hiện Dị thường (Anomaly Detection & Decision Logic)

#### Tính toán Anomaly Score
Sai số dự báo (Residual MSE) đóng vai trò là điểm dị thường $A(t)$ tại thời điểm $t$:

$$
A(t) = \frac{1}{C \cdot H} \sum_{c=1}^C \sum_{h=1}^H (y_{\text{true}, c, h}(t) - y_{\text{forecast}, c, h}(t))^2
$$

#### Hiệu chuẩn Ngưỡng động POT (Peak-Over-Threshold Calibration)
Dựa trên Thuyết Giá trị Cực trị (Extreme Value Theory - EVT), ta mô hình hóa phần đuôi của phân phối sai số trên tập lành mạnh:
1. Xác định một ngưỡng cơ sở $u$ sao cho các phần vượt ngưỡng $A(t) - u$ tuân theo Phân phối Pareto Tổng quát (Generalized Pareto Distribution - GPD).
2. Ước lượng tham số hình dáng $\xi$ và tỷ lệ $\sigma$ của GPD qua phương pháp MLE.
3. Tính toán ngưỡng quyết định động $z_q$ cho xác suất vượt ngưỡng mục tiêu cực thấp $q = 10^{-3}$:

   $$
   z_q \approx u + \frac{\sigma}{\xi} \left( \left(\frac{N}{N_u} q\right)^{-\xi} - 1 \right)
   $$
Nếu $A(t) > z_q$, hệ thống sẽ kích hoạt cảnh báo bất thường.
