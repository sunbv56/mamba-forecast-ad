# 📋 Checklist Chuẩn Bị Viết Bài Báo Khoa Học (Mamba-Forecast-AD)

Tài liệu này cung cấp danh sách kiểm tra chi tiết các đầu việc kỹ thuật và các quy tắc hành văn khoa học cần tuân thủ để phục vụ cho việc viết bài báo nghiên cứu đạt chuẩn Q1.

---

## 1. ⚙️ Trạng Thái Git & Đồng Bộ Mã Nguồn (Git & Source Code Verification)

Đảm bảo mã nguồn huấn luyện, kiểm thử và đánh giá đã được tách biệt hoàn toàn từ Jupyter Notebooks và được quản lý chặt chẽ trong Git.

- [x] **Mô-đun hóa mã nguồn**: Kiểm tra xem toàn bộ các thành phần của hệ thống đã được phân tách từ các file Notebook (`final-eval-mamba-forecast-ad.ipynb` và `final-mamba-forecast-ad.ipynb`) vào thư mục `src/` chưa.
  - [x] **Xử lý dữ liệu (`src/data/`)**: Đã có `MultiBearingDataset` hỗ trợ lọc thông dải (high-pass filter), tính toán đặc trưng vật lý (Stats Head) và chia tập dữ liệu huấn luyện/kiểm thử theo thời gian.
  - [x] **Kiến trúc mô hình (`src/models/`)**: Đã chứa mô hình đề xuất `HybridMambaCNN` và các baselines so sánh (`LSTM`, `PatchLSTM`, `ModernTCN`, `PatchTST`, `MambaTSOfficial`).
  - [x] **Huấn luyện (`src/training/`)**: Đã chứa `train.py` thực hiện vòng lặp huấn luyện với cơ chế tự động co giãn tham số (auto-scaling) và dừng sớm (`EarlyStopping`).
  - [x] **Đánh giá (`src/evaluation/`)**: Đã chứa `eval.py` thực hiện đánh giá đa mô hình trên tập kiểm thử vòng bi độc lập và các giải thuật tính ngưỡng tự học (`metrics.py`, `anomaly_scorer.py`).
- [x] **So sánh tính năng giữa Script và Notebook**:
  - Mã nguồn trong `src/` đã đồng bộ hoàn chỉnh logic cốt lõi của bài báo.
  - *Điểm khác biệt nhỏ*: Notebook đánh giá `final-eval-mamba-forecast-ad.ipynb` chứa thêm phần giám sát tài nguyên phần cứng chi tiết (Peak GPU VRAM) và phân rã độ trễ thời gian thực (Data Transfer, Inference, Anomaly Scoring, Decision Latency). Các số liệu đo đạc này sẽ được trích xuất trực tiếp để đưa vào phần thực nghiệm của bài báo.

---

## 2. 📝 Các Tài Liệu Hướng Dẫn Kèm Theo (Documentation Deliverables)

Tạo và hoàn thiện 3 tài liệu hướng dẫn kỹ thuật cốt lõi tại thư mục gốc của dự án:

- [x] **`running_guide.md` (Hướng dẫn chạy chương trình)**: Hướng dẫn chi tiết từ bước cài đặt môi trường CUDA, đồng bộ hóa tập dữ liệu thông qua Hugging Face Hub, đến việc chạy huấn luyện và đánh giá mô hình bằng dòng lệnh.
- [x] **`configuration_guide.md` (Hướng dẫn cấu hình)**: Hướng dẫn chi tiết ý nghĩa các tham số trong file cấu hình YAML, giải thích cơ chế tắt RevIN và Multi-scale patching trong phiên bản cuối cùng, cấu trúc Stats Head, và thiết lập các thí nghiệm triệt tiêu (Ablation Study).
- [x] **`source_code_description.md` (Mô tả mã nguồn)**: Mô tả kiến trúc thư mục dự án, luồng dữ liệu của hệ thống và giải thích chức năng của từng lớp/mô-đun trong mã nguồn.

---

## 3. ✍️ Quy Tắc Viết Bài Báo Khoa Học (Paper Writing Guidelines & Constraints)

Khi bắt tay vào viết bản thảo bài báo (Manuscript), cần tuân thủ nghiêm ngặt các quy định học thuật sau:

- [ ] **Giới hạn độ dài**: Bài báo chỉ viết ngắn gọn, súc tích trong phạm vi **15 - 18 trang** (bao gồm cả tài liệu tham khảo và hình vẽ). Tập trung tối đa vào đóng góp học thuật và kết quả thực nghiệm thay vì viết lan man.
- [ ] **Sử dụng thể bị động (Passive Voice)**: Trình bày toàn bộ nội dung nghiên cứu bằng thể bị động để đảm bảo tính khách quan khoa học.
- [ ] **Tuyệt đối không sử dụng đại từ ngôi thứ nhất/chủ động**:
  - **CẤM** các cụm từ như: *We propose* (Chúng tôi đề xuất), *Our model* (Mô hình của chúng tôi), *Our system* (Hệ thống của chúng tôi), *We evaluated* (Chúng tôi đã đánh giá), *In this paper, we...*
  - **Bảng quy đổi ví dụ cụ thể để tránh lỗi**:
    
    | ❌ Kiểu viết bị cấm (Active Voice) |  Kiểu viết chuẩn khoa học (Passive Voice) |
    | :--- | :--- |
    | **We propose** a hybrid Mamba-CNN architecture... | **A hybrid Mamba-CNN architecture is proposed**... |
    | **Our model** outperforms the baselines... | **The proposed model outperforms** the baselines... |
    | **Our system** consists of three stages... | **The system is composed of** three stages... |
    | **We evaluate** the models on the B02 dataset... | **The models are evaluated** on the B02 dataset... |
    | **We calculate** the anomaly score using MSE... | **The anomaly score is computed** via Mean Squared Error (MSE)... |
    | **We choose** a lookback window of 4096... | **A lookback window of 4096 is selected**... |
    | **We train** the model for 10 epochs... | **The model is trained** for 10 epochs... |
    | **We disable** RevIN to avoid leakage... | **RevIN is disabled** to prevent information leakage... |

---

## 4. 🔍 Kiểm Tra Tính Đúng Đắn Khoa Học (Scientific Rigor & Correctness)

Trước khi viết phần Phương pháp và Thực nghiệm, hãy kiểm tra và xác nhận các luận điểm khoa học sau để phản biện tốt với Reviewers Q1:

- [ ] **Lý giải tắt RevIN & Multi-scale Patching**:
  - *RevIN (Reversible Instance Normalization)*: Bị vô hiệu hóa (`use_revin: false`) do việc chuẩn hóa trung bình/phương sai cục bộ trên từng cửa sổ làm mất đi đặc trưng tiến trình suy giảm (degradation trend) dài hạn của vòng bi. Điều này khiến điểm bất thường ở giai đoạn cuối bị kéo thấp xuống, dẫn đến lỗi bỏ sót cảnh báo.
  - *Multi-scale patching*: Bị tắt để giữ mô hình tinh gọn, tránh overfitting trên tập dữ liệu nhỏ và giảm độ trễ tính toán trong ứng dụng giám sát thời gian thực.
- [ ] **Stats Head (Vật lý dẫn đường - Physics-Informed)**:
  - Khẳng định mô hình nhận đầu vào là chuỗi thời gian thô và được bổ trợ bởi 8 đặc trưng thống kê vật lý (RMS, Kurtosis, Crest Factor, Shape Factor, Impulse Factor, Margin Factor, Peak-to-Peak, Variance) thông qua Stats Head. Cấu trúc này hướng mô hình tập trung vào sự biến đổi cơ học thực tế thay vì chỉ học các đặc trưng số học thuần túy.
- [ ] **Hiệu chuẩn ngưỡng không rò rỉ dữ liệu (Leakage-Free Calibration)**:
  - Ngưỡng phát hiện bất thường (POT, 3-Sigma, Robust) được tính toán cục bộ cho từng vòng bi (Per-bearing calibration) dựa trên dữ liệu khỏe mạnh ở giai đoạn đầu (`bearing_labels == 0` trong khoảng `skip_ratio` đến `train_ratio`). Cách tiếp cận này đảm bảo mô hình tự thích ứng với từng thiết bị riêng biệt mà không sử dụng trước bất kỳ thông tin nào từ tập kiểm thử lỗi, phản ánh chính xác quy trình triển khai công nghiệp thực tế.
