# BÁO CÁO TIẾN ĐỘ DỰ ÁN: MAMBA-FORECAST-AD
**Chủ đề:** Kiến trúc lai Mamba-CNN kết hợp đặc trưng vật lý cho dự báo và chẩn đoán hư hỏng vòng bi.

---

### 1. Tổng quan dự án (Giới thiệu)
*   **Mục tiêu:** Xây dựng hệ thống dự báo chuỗi thời gian và phát hiện bất thường dựa trên dữ liệu rung động tần số cao (Dataset B02).
*   **Vấn đề giải quyết:** 
    *   Transformer truyền thống có chi phí tính toán cực lớn (O(N²)) với dữ liệu chuỗi dài.
    *   CNN đơn thuần khó bắt được các phụ thuộc xa trong quá trình thoái hóa vòng bi.
    *   Dữ liệu thực tế thường không dừng (non-stationary), gây khó khăn cho việc hội tụ của mô hình.

### 2. Phương pháp đề xuất (Kiến trúc CI-Mamba++)
Mô hình hoạt động dựa trên cơ chế lai ghép tối ưu giữa học sâu hiện đại và tri thức vật lý:
*   **RevIN (Kim et al., 2021):** Tự động chuẩn hóa và giải chuẩn hóa để xử lý hiện tượng trôi phân phối (distribution shift) khi vòng bi hỏng dần theo thời gian.
*   **Series Decomposition (Zeng et al., 2023 - DLinear):** Tách tín hiệu rung động thành hai nhánh: **Trend** (xu hướng thoái hóa dài hạn) và **Seasonal** (các dao động chu kỳ ngắn).
*   **Multi-scale Patching (Nie et al., 2022 - PatchTST):** Sử dụng các cửa sổ trượt nhiều kích thước để trích xuất đặc trưng cục bộ tần số cao và giảm nhiễu tín hiệu trước khi đưa vào backbone.
*   **Mamba Backbone (Gu & Dao, 2023):** Đóng vai trò bộ xử lý trung tâm, học các phụ thuộc thời gian dài với độ phức tạp tuyến tính, giúp mô hình cực nhanh và nhẹ hơn Transformer.
*   **Physical Feature Fusion (Physics-informed AI):** Tích hợp trực tiếp 8 chỉ số thống kê đặc trưng của bài toán chẩn đoán lỗi (RMS, Kurtosis, Skewness,...) vào đầu ra để tăng độ nhạy với các lỗi vật lý thực tế.

### 3. Các bước thực hiện (Chi tiết Pipeline)
Quy trình được thiết kế theo luồng xử lý **Hybrid Pipeline**, kết hợp cấu trúc phân tách tín hiệu của **DLinear (2023)**, cơ chế phân mảnh của **PatchTST (2022)** và logic phát hiện bất thường dựa trên sai số dự báo (Forecasting-based AD):

<!-- <img src="./CI-Mamba++%20Processing.png" alt="Sơ đồ quy trình CI-Mamba++" style="max-height: 23cm; display: block; margin: 0 auto;"> -->

*   **Bước 1: Tiền xử lý dữ liệu thô (Raw Data to Windows):**
    *   Tiếp nhận tín hiệu rung động thô từ cảm biến (Sampling rate 128kHz) dưới dạng file `.mat`.
    *   Sử dụng kỹ thuật **Sliding Window** để chia nhỏ tín hiệu thành các đoạn dữ liệu (Lookback segments) có độ dài cố định (ví dụ: 2048 điểm).
    *   Đồng thời tính toán bộ 8 đặc trưng vật lý (RMS, Kurtosis, Skewness, Peak-to-Peak,...) làm đầu vào bổ trợ.

*   **Bước 2: Chuẩn hóa và Phân tách tín hiệu (Preprocessing & Decomposition):**
    *   **RevIN (Normalization):** Chuẩn hóa thích nghi cho từng đoạn dữ liệu để loại bỏ sự khác biệt về biên độ do điều kiện vận hành thay đổi.
    *   **Series Decomposition:** Tách đoạn tín hiệu đã chuẩn hóa thành hai thành phần độc lập: **Trend** (xu hướng thoái hóa dài hạn) và **Seasonal** (các dao động chu kỳ và xung động lỗi).

*   **Bước 3: Trích xuất đặc trưng và Học chuỗi (Feature Learning):**
    *   Thành phần Seasonal được đưa qua **Multi-scale Patching** để nhóm dữ liệu thành các mảnh (tokens) ở nhiều quy mô thời gian khác nhau.
    *   **Mamba Encoder** thực hiện xử lý tuần tự các mảnh này để bắt được các phụ thuộc thời gian cực dài – dấu hiệu của sự hư hỏng chớm nở.

*   **Bước 4: Hợp nhất tri thức và Dự báo (Fusion & Forecasting):**
    *   **Feature Fusion:** Kết hợp đặc trưng học sâu (từ Mamba) với đặc trưng vật lý (từ Bước 1) thông qua một Fusion Head.
    *   Nhánh Trend và Seasonal sau đó được dự báo riêng biệt và tổng hợp lại để tái tạo tín hiệu tương lai (**Output Forecast**).

*   **Bước 5: Chẩn đoán và Phát hiện bất thường (Detection & Diagnosis):**
    *   Tính toán sai số dự báo (Residuals) giữa tín hiệu thực tế và kết quả từ mô hình.
    *   Áp dụng thuật toán **POT (Peak Over Threshold)** hoặc ngưỡng **3-Sigma** lên sai số này để xác định trạng thái sức khỏe vòng bi và đưa ra cảnh báo sớm.

### 4. Cơ sở lý thuyết
Dự án được xây dựng dựa trên sự kết hợp giữa các kiến trúc học sâu tiên tiến (Mamba, Transformer) và tri thức vật lý chuyên ngành:

*   **Multi-scale Patching & Feature Extraction:**
    *   Kế thừa kỹ thuật **Patching** từ **PatchTST (Nie et al., 2022)** và **Patch-Mamba (2024)** để giảm số lượng token xử lý, đồng thời tăng khả năng học ngữ nghĩa cục bộ của tín hiệu.
    *   Cơ chế **Multi-scale** lấy cảm hứng từ **TimesNet (Wu et al., 2023)** và **MS-Mamba (2024)**, giúp mô hình bắt được các dao động ở nhiều chu kỳ và độ phân giải khác nhau của dữ liệu rung động.

*   **Series Decomposition (Phân tách chuỗi):**
    *   Áp dụng tư tưởng từ **DLinear (Zeng et al., 2023)** và **Autoformer (Wu et al., 2021)**. Việc tách Trend/Seasonal giúp mô hình hóa tường minh xu hướng thoái hóa dài hạn và các xung động lỗi ngắn hạn, khắc phục điểm yếu của các kiến trúc "hộp đen" truyền thống.

*   **RevIN (Normalization - Chuẩn hóa):**
    *   Sử dụng giải pháp từ bài báo **Kim et al. (2021)** để xử lý vấn đề dữ liệu không dừng (non-stationary). Đây là kỹ thuật then chốt giúp các phiên bản Transformer và Mamba ổn định khi đối mặt với sự thay đổi phân phối dữ liệu (distribution shift) trong quá trình vòng bi hư hỏng.

*   **Feature Fusion & Physical Knowledge (8 đặc trưng vật lý):**
    *   Tích hợp tri thức chuyên ngành (**Physics-informed AI**) dựa trên các nghiên cứu kinh điển về chẩn đoán lỗi máy móc (**Lei et al., 2018**). Việc trích xuất 8 chỉ số (RMS, Kurtosis, Skewness,...) và kết hợp chúng với Deep Features thông qua cơ chế **Fusion Head** giúp mô hình bám sát các tiêu chuẩn kỹ thuật thực tế.

*   **Mamba Backbone & CI Architecture:**
    *   Dựa trên nghiên cứu gốc **Mamba (Gu & Dao, 2023)** và tư duy **Channel-Independent (iTransformer - Liu et al., 2023)**. Sự kết hợp này cho phép xử lý dữ liệu cảm biến đa kênh với độ phức tạp tuyến tính nhưng vẫn giữ được độ chính xác tương đương hoặc cao hơn các mô hình Transformer cồng kềnh.

*   **Lưu ý về đặc trưng Nhiệt độ (Temperature Features):**
    *   **Chỉ số trễ (Lagging Indicator):** Dựa trên các nghiên cứu về cơ học (**Lei et al., 2018**), nhiệt độ thường là kết quả của ma sát sau khi hư hỏng đã xảy ra nghiêm trọng. Việc sử dụng nhiệt độ có thể khiến mô hình phản ứng chậm hơn so với việc chỉ tập trung vào tín hiệu rung động – vốn bắt được các vết nứt vi mô ngay từ giai đoạn khởi đầu.
    *   **Sự lệch pha tần số (Frequency Mismatch):** Có sự khác biệt cực lớn về tốc độ lấy mẫu (Vibration lấy mẫu ở kHz, trong khi Temperature thường ở Hz). Việc kết hợp trực tiếp hai loại dữ liệu này dễ gây nhiễu cho cơ chế Selective Scan của Mamba, khiến mô hình bị "đánh lừa" bởi các chu kỳ thay đổi chậm của môi trường thay vì các pattern lỗi tần số cao.
    *   **Rủi ro Overfitting:** Nhiệt độ chịu ảnh hưởng lớn từ môi trường bên ngoài (nhiệt độ phòng, tải trọng vận hành) không liên quan đến lỗi, dẫn đến nguy cơ mô hình học các tương quan giả (spurious correlations), làm giảm khả năng tổng quát hóa trên dữ liệu thực tế.

### 5. Định hướng cải thiện (Future Work)
Dựa trên việc tham khảo các Pipeline mới nhất từ các model Mamba Forecasting và Anomaly Detection hiện nay, dự án có thể được nâng cấp qua các hướng sau:

*   **Nâng cấp lên Mamba-2 (Structured State Space Duality):** Sử dụng phiên bản Mamba-2 mới nhất (2024) để tận dụng cơ chế SSD, giúp tăng tốc độ huấn luyện và khả năng biểu diễn các đặc trưng phi tuyến phức tạp hơn.
*   **Modeling tương quan đa kênh (Cross-channel Correlations):** Hiện tại mô hình đang chạy ở chế độ Channel-Independent (CI). Việc tích hợp thêm cơ chế học tương quan giữa các cảm biến (ví dụ: kết hợp rung động trục X, Y, Z) thông qua **iTransformer-style attention** hoặc **Mamba-ND** sẽ giúp bắt được các lỗi có tính đa hướng.
*   **Phân tích trong miền Tần số (Frequency Domain Integration):** Áp dụng biến đổi Fourier (FFT) hoặc Wavelet ngay trong khối Mamba (tương tự ý tưởng của **TimesNet** hoặc **FreTS**) để mô hình hóa trực tiếp các đặc trưng tần số đặc thù của lỗi vòng bi (như tần số vòng trong, vòng ngoài).
*   **Cơ chế Loss đa nhiệm (Multi-task Learning):** Kết hợp đồng thời bài toán **Dự báo (Forecasting)** và **Tái tạo (Reconstruction)**. Việc này giúp mô hình học được biểu diễn không gian trạng thái (latent space) tốt hơn, từ đó tăng độ nhạy khi phát hiện các bất thường nhỏ nhất.
*   **Decomposition nâng cao (Advanced Decomposition):** Thay thế cơ chế Moving Average đơn giản bằng các kỹ thuật phân tách tín hiệu mạnh mẽ hơn như **Wavelet Decomposition** hoặc **EMD (Empirical Mode Decomposition)** để tách biệt hoàn toàn nhiễu môi trường khỏi dấu hiệu hư hỏng.

---