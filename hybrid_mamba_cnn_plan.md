# Kế hoạch Triển khai: Sinh viên AI 2 (Code beta, Experiment automation & Visualization)

## 1. Mục tiêu và Nhiệm vụ chính
Vai trò của Sinh viên AI 2 tập trung vào việc **Dựng source code beta**, **Xây dựng pipeline thử nghiệm tự động** và **Trực quan hóa kết quả**. 

### 1.1. Dựng Pipeline Dữ liệu và Mô hình (Code Beta)
- **Tạo Pipeline Data**:
  - **Load dataset**: Đọc dữ liệu thô.
  - **Windowing**: Cắt dữ liệu thành các cửa sổ trượt.
  - **Normalize**: Chuẩn hóa dữ liệu đầu vào.
  - **Build forecasting pairs**: Tạo cặp dữ liệu đầu vào (chuỗi quá khứ $X$) và mục tiêu (chuỗi tương lai $Y$).
- **Huấn luyện (Training)**:
  - **Unsupervised Learning**: Huấn luyện mô hình chỉ trên dữ liệu trạng thái bình thường (healthy data) để học đặc trưng ổn định.
  - **Evaluate anomaly score**: Đánh giá điểm số bất thường (anomaly score) dựa trên sai số dự báo (Forecasting MSE). Sai số này đóng vai trò như một Health Index tự nhiên.
  - **Draw chart**: Tích hợp các hàm vẽ biểu đồ trong quá trình đánh giá.

### 1.2. Chạy Baseline
Triển khai và huấn luyện các mô hình cơ sở để làm hệ quy chiếu:
- **LSTM**: Mô hình tuần tự truyền thống.
- **TCN (Temporal Convolutional Network)**: Mạng tích chập thời gian.
- **Transformer nhỏ**: Đánh giá cơ chế Attention cơ bản.
- **Mamba (nếu kịp)**: Tích hợp kiến trúc Mamba để kiểm chứng hiệu năng so với các baseline.

### 1.3. Tự động hóa Thực nghiệm (Experiment Automation)
Tạo script `auto-run` để có thể tự động chạy lưới (grid search) qua nhiều cấu hình siêu tham số quan trọng:
- **lookback N**: Số bước thời gian quan sát trong quá khứ.
- **horizon K**: Số bước thời gian cần dự báo trong tương lai.
- **patch size**: Kích thước gộp dữ liệu (nếu áp dụng patching).
- Các siêu tham số khác: **learning rate**, **batch size**, **hidden dimension**.

### 1.4. Xuất Đầu ra (Outputs & Visualization)
Hệ thống phải xuất các kết quả sau một cách tự động và rõ ràng:
- **CSV log**: Nhật ký thử nghiệm chứa thông tin cấu hình và kết quả.
- **Confusion matrix**: Ma trận nhầm lẫn cho bài toán phát hiện lỗi.
- **Biểu đồ Anomaly score over TTF%**: Trực quan hóa điểm số bất thường theo phần trăm thời gian đến khi hỏng (Time-to-Failure).
- **Threshold chart**: Biểu đồ hiển thị ngưỡng phân định bất thường.
- **Các chỉ số khác**: Detection delay (độ trễ phát hiện), False alarm rate (tỷ lệ báo động giả).

## 2. Câu hỏi nghiên cứu hỗ trợ (Research Questions)
Các công việc trên nhằm cung cấp thực nghiệm và dữ liệu trực tiếp giải quyết 4 câu hỏi nghiên cứu sau:
- **SQ5**: Cấu hình `lookback N` và `forecast horizon K` nào cho Anomaly Score phân tách rõ ràng nhất giữa trạng thái bình thường và bất thường?
- **SQ6**: Mô hình nào trong số các baseline (và Mamba) chạy nhanh nhất (inference/train time) nhưng vẫn giữ được chỉ số F1/AUC ở mức tốt?
- **SQ7**: Kiến trúc Mamba có thực sự tiết kiệm bộ nhớ và thời gian hơn Transformer khi huấn luyện trên chuỗi dữ liệu dài không?
- **SQ8**: Anomaly Score bắt đầu tăng vọt ở giai đoạn nào của TTF%? Tại sao sai số dự báo MSE lại là một chỉ báo (indicator) nhạy bén cho sự suy thoái ngay cả khi chưa có lỗi rõ rệt?
- **SQ9**: Làm thế nào để thiết lập ngưỡng (threshold) chống chịu được nhiễu trong tập huấn luyện? So sánh hiệu quả giữa 3-sigma và Robust Thresholding (Median/IQR) trong việc giảm tỷ lệ báo động giả (FAR).

## 3. Đầu ra yêu cầu (Deliverables)
Để hoàn thành giai đoạn này, Sinh viên AI 2 cần bàn giao:
1. **Code beta**: Hoàn thiện pipeline, models, vòng lặp train/test.
2. **Experiment log**: File nhật ký lưu lại kết quả của nhiều lần chạy.
3. **Biểu đồ kết quả**: Hình ảnh trực quan (Threshold chart, Anomaly over TTF%).
4. **Bảng so sánh baseline**: Đánh giá tương quan hiệu năng giữa LSTM, TCN, Transformer nhỏ và Mamba.
5. **Script tự động**: Công cụ để chạy benchmark hàng loạt nhiều cấu hình.
