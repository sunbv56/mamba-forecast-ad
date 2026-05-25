# ⚙️ Hướng Dẫn Cấu Hình Mô Hình (Mamba-Forecast-AD Configuration Guide)

Tài liệu này giải thích ý nghĩa các tham số cấu hình trong thư mục `configs/` và cung cấp hướng dẫn thiết lập tối ưu để đạt kết quả tốt nhất trên tập dữ liệu vòng bi Paderborn (B02).

---

## 1. 📂 Cấu Trúc File Cấu Hình YAML

Các file cấu hình YAML được chia làm 4 nhóm chính: `data` (dữ liệu), `model` (mô hình), `training` (huấn luyện), và `logging` (nhật ký). 

Dưới đây là ví dụ cấu hình chuẩn cho mô hình lai Mamba-CNN:

```yaml
data:
  raw_dir: "data/raw/B04"
  processed_dir: "data/processed/B04"
  train_datasets: ["data/processed/B02", "data/processed/B05"]
  test_datasets: ["data/processed/B02", "data/processed/B03", "data/processed/B04", "data/processed/B05"]
  sampling_rate: 128000
  highpass_freq: 2000
  label_strategy: 'rms'
  window_stride: 1024
  lookback: 4096
  horizon: 1024
  skip_ratio: 0.05
  train_ratio: 0.4

model:
  patch_size: 64  
  patch_stride: 32
  trend_downsample: 64
  cnn_out_channels: 64
  mamba_d_model: 64
  mamba_n_layer: 4 
  mamba_d_state: 16
  mamba_d_conv: 3
  mamba_expand: 3
  bidirectional: false
  decomp_kernel: 25 
  auto_scale_baselines: true
  use_decomposition: true
  use_stats: true

training:
  batch_size: 128
  learning_rate: 5e-4
  epochs: 10
  device: "cuda"
```

---

## 2. 🔍 Giải Thích Các Tham Số Cốt Lõi & Hướng Dẫn Thiết Lập Tối Ưu

### A. Nhóm Dữ Liệu (`data`)

- **`highpass_freq` (Mặc định: `2000` Hz)**: Lọc thông cao cho tín hiệu rung thô.
  - *Ý nghĩa vật lý*: Giúp loại bỏ các thành phần nhiễu tần số thấp từ động cơ nền và làm nổi bật các xung va đập cơ học tần số cao do vết nứt vòng bi gây ra.
- **`lookback` (Ví dụ: `4096`) & `horizon` (Ví dụ: `1024`)**: 
  - *Độ dài chuỗi lịch sử (`lookback`)*: Cần đủ dài để bao quát tối thiểu 2-3 chu kỳ quay của vòng bi (thường từ 2048 đến 4096 mẫu với tần số lấy mẫu đã hạ thấp).
  - *Độ dài chuỗi dự báo (`horizon`)*: Chọn ở mức vừa phải (128 đến 1024). Dự báo quá dài sẽ làm giảm độ chính xác, dự báo quá ngắn sẽ không đủ thông tin để tính Anomaly Score ổn định.
- **`train_ratio` (Mặc định: `0.4` hoặc `0.5`)**: Tỷ lệ mẫu ở đầu chu kỳ sống dùng để huấn luyện.
  - *Quy tắc*: Chỉ huấn luyện trên giai đoạn hoạt động khỏe mạnh ban đầu (Healthy State). Tránh đặt `train_ratio` quá lớn (>0.6) vì có thể đưa tín hiệu suy thoái ban đầu vào tập huấn luyện, làm mô hình học cả trạng thái lỗi.
- **`skip_ratio` (Mặc định: `0.05`)**: Bỏ qua 5% dữ liệu ban đầu.
  - *Lý do*: Giai đoạn bắt đầu hoạt động (run-in period) thường có rung động không ổn định do thiết bị đang rà khớp, dễ gây nhiễu cho mô hình.

---

### B. Nhóm Mô Hình & Cơ Chế Thực Nghiệm Tối Ưu (`model`)

- **`auto_scale_baselines` (`true`)**: Tự động co giãn tham số baselines.
  - *Ý nghĩa*: Khi đặt là `true`, script huấn luyện sẽ tự động tìm kiếm và điều chỉnh số chiều ẩn (`hidden_dim`, `d_model`) của các mô hình đối chứng (LSTM, PatchLSTM, ModernTCN, PatchTST) sao cho tổng số lượng tham số học tập của chúng tương đương với mô hình lai Mamba-CNN (~200k - 300k tham số). Điều này đảm bảo sự **so sánh công bằng tuyệt đối** về dung lượng tính toán (Fair Parameter Budget).
- **Loại bỏ RevIN và Chuẩn hóa z-score tức thời (Instance Normalization)**:
  - *Phân tích khoa học*: RevIN và chuẩn hóa z-score trên từng cửa sổ (instance normalization) **đã bị xóa bỏ hoàn toàn khỏi mã nguồn** thay vì duy trì dưới dạng cấu hình bật/tắt. Cơ sở khoa học là các cơ chế chuẩn hóa này thực hiện trừ đi trung bình (mean) và chia cho độ lệch chuẩn (std) cục bộ của từng cửa sổ. Khi vòng bi bước vào giai đoạn suy thoái nghiêm trọng, biên độ rung thực tế (RMS) tăng vọt. Việc chuẩn hóa tức thời vô tình co kéo biên độ lỗi về mức bình thường giống hệt chuỗi khỏe mạnh, triệt tiêu tín hiệu suy thoái và làm sai số dự báo (Anomaly Score) không tăng ở cuối đời máy. **Biên độ vật lý tuyệt đối phải được bảo toàn nguyên vẹn.**
- **Loại bỏ phân nhánh đa tỷ xích (Multi-scale Patching)**:
  - *Lý do*: Cơ chế đa tỷ xích đã bị loại bỏ khỏi kiến trúc mô hình vì các thực nghiệm thực tế cho thấy nó làm tăng nguy cơ quá khớp (overfitting) trên tập dữ liệu kích thước trung bình và tăng đáng kể chi phí/độ trễ tính toán không cần thiết.
- **`use_stats: true` (Stats Head - Hướng dẫn vật lý)**: Kích hoạt đầu Stats Head.
  - *Ý nghĩa*: Stats Head trích xuất 8 đặc trưng thống kê miền thời gian (RMS, Kurtosis, Crest Factor, Shape Factor, Impulse Factor, Margin Factor, Peak-to-Peak, Variance) từ cửa sổ lookback và đưa vào làm đặc trưng bổ trợ. Điều này giúp hướng dẫn mô hình bằng tri thức vật lý cơ học dòng máy, cải thiện đáng kể độ chính xác so với việc chỉ học chuỗi thời gian thuần túy.

---

### C. Nhóm Huấn Luyện (`training`)

- **`loss` (`HuberLoss`)**: Sử dụng tổn thất Huber thay vì MSE thuần túy trong quá trình huấn luyện.
  - *Lý do*: Tín hiệu rung cơ học trong môi trường công nghiệp thường chứa các xung gai đột biến (outliers) do va đập ngẫu nhiên không phải lỗi. HuberLoss hoạt động như L1-loss với các lỗi lớn (ít bị ảnh hưởng bởi outliers) và hoạt động như L2-loss với các lỗi nhỏ (giúp hội tụ mịn).
- **`scheduler` (`CosineAnnealingLR`)**: Giảm tốc độ học theo hàm Cosine.
  - *Ý nghĩa*: Giúp mô hình lai Mamba-CNN dễ dàng thoát khỏi các cực tiểu cục bộ ở giai đoạn đầu và hội tụ ổn định ở giai đoạn cuối của quá trình tối ưu hóa.
