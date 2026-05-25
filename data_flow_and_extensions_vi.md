# Sơ đồ Luồng Dữ liệu và Định hướng Phát triển Kiến trúc CI-Mamba++

Tài liệu này trình bày trực quan và chi tiết quy trình xử lý dữ liệu hiện tại từ tín hiệu rung thô đến đầu ra cảnh báo, đồng thời đề xuất các hướng phát triển nâng cao (Physics-Informed & Domain Adaptation) để tối ưu hóa mô hình phục vụ báo cáo khoa học.

---

## 1. Sơ đồ Luồng Dữ liệu Hiện tại (Current Data Flow)

Sơ đồ dưới đây mô tả quá trình xử lý tín hiệu rung thô trực tiếp (không qua chuẩn hóa Z-score tín hiệu hay RevIN), tính toán các đặc trưng vật lý tĩnh, phân tách chuỗi thời gian, huấn luyện mạng Mamba kênh độc lập (Channel-Independent), dung hợp đặc trưng vật lý tại đầu ra và đưa ra quyết định cảnh báo thông qua thuật toán POT (Peak-Over-Threshold).

```mermaid
graph TD
    %% Định nghĩa Style
    classDef raw fill:#f9e6ff,stroke:#8a2be2,stroke-width:2px;
    classDef process fill:#e6f2ff,stroke:#0066cc,stroke-width:1px;
    classDef stats fill:#ffe6e6,stroke:#cc0000,stroke-width:2px;
    classDef model fill:#e6ffe6,stroke:#009933,stroke-width:2px;
    classDef output fill:#fff9e6,stroke:#cc9900,stroke-width:2px;

    %% 1. Raw Data Stage
    subgraph "1. TIỀN XỬ LÝ DỮ LIỆU THÔ"
        A["Tín hiệu rung thô (.pt)<br>Gia tốc trục ngang & dọc<br>Kích thước: (C=2, L_total)"] -->|Cửa sổ trượt| B["Cửa sổ Lookback (x_raw)<br>Kích thước: (C=2, L=1024)"]
        B -->|Gán nhãn dựa trên RMS| Label["Nhãn mục tiêu (Label)<br>0: Khỏe mạnh / 1: Lỗi"]
    end
    
    %% 2. Processing & Normalization
    subgraph "2. TRÍCH XUẤT ĐẶC TRƯNG VẬT LÝ"
        B -->|Tính 8 Stats trên tín hiệu thô| C["Bộ 8 đặc trưng vật lý (stats)<br>RMS, Kurtosis, Skewness,...<br>Kích thước: (C=2, 8)"]
    end

    %% 3. Model Stage
    subgraph "3. KIẾN TRÚC MÔ HÌNH (CI-MAMBA++)"
        %% Signal flows directly to Decomposition without Z-Score
        B --> E["Phân rã chuỗi (Series Decomposition)"]
        E -->|Nhánh Seasonal| F["Thành phần dao động (x_seas)<br>Kích thước: (C=2, L)"]
        E -->|Nhánh Trend| G["Thành phần xu hướng (x_trend)<br>Kích thước: (C=2, L)"]
        
        %% Seasonal processing
        F -->|Phân mảnh - Patching| H["Nhúng mảnh tích chập<br>Kích thước: (C=2, N, D_model)"]
        H -->|Gộp kênh - Fold CI| I["Gộp Batch & Channel (CI)<br>Kích thước: (B*C, N, D_model)"]
        I -->|Mamba Encoder| J["Vector ẩn (s_hidden)<br>Kích thước: (B*C, N, D_model)"]
        J -->|Average & Max Pooling| K["Đặc trưng nén (s_flat)<br>Kích thước: (B*C, D_model)"]
        
        %% Stats fusion with BatchNorm1d in model
        K --> Fusion{"Fusion Forecast Head<br>(Đầu dự báo liên kết)"}
        C -->|Gộp kênh - Fold CI| L["Folded Stats<br>Kích thước: (B*C, 8)"]
        L -->|BatchNorm1d trong Model| StatsNorm["Stats chuẩn hóa<br>Kích thước: (B*C, 8)"]
        StatsNorm --> Fusion
        
        Fusion -->|Phép chiếu Tuyến tính| M["Dự báo gộp (y_seas_folded)<br>Kích thước: (B*C, H=64)"]
        M -->|Giải gộp kênh - Unfold CI| N["Dự báo Seasonal (y_seasonal)<br>Kích thước: (B, C, H=64)"]
        
        %% Trend processing
        G -->|AvgPool1d + Tuyến tính nhẹ| O["Dự báo Trend (y_trend)<br>Kích thước: (B, C, H=64)"]
    end

    %% 4. Output Stage
    subgraph "4. ĐẦU RA & QUYẾT ĐỊNH CHẨN ĐOÁN"
        N & O -->|Trộn thích ứng - alpha học được| P["Tín hiệu dự báo thực (y_pred)<br>Kích thước: (B, C, H=64)"]
        P -->|Tính sai số bình phương MSE| R["Điểm bất thường (Anomaly Score = MSE)"]
        R -->|Ngưỡng động cực trị POT| Final["Quyết định: Bình thường (0) / Cảnh báo (1)"]
    end

    class A,B raw;
    class E,F,G,H,I,J,K process;
    class C,L,StatsNorm stats;
    class J,Fusion,O,P model;
    class R,Final,Label output;
```


### 1.1. Công thức Toán học của bộ 8 chỉ số vật lý (`stats`)
Được tính toán trực tiếp trên tín hiệu rung thô chưa chuẩn hóa $x \in \mathbb{R}^{L}$:
1. **Mean (Trung bình):**
   $$\mu = \frac{1}{L} \sum_{i=1}^L x_i$$
2. **Standard Deviation (Độ lệch chuẩn):**
   $$\sigma = \sqrt{\frac{1}{L} \sum_{i=1}^L (x_i - \mu)^2} + \epsilon$$
3. **RMS (Trị hiệu dụng):**
   $$\text{RMS} = \sqrt{\frac{1}{L} \sum_{i=1}^L x_i^2}$$
4. **Peak-to-Peak (Đỉnh - Đỉnh):**
   $$X_{p-p} = \max(x) - \min(x)$$
5. **Skewness (Độ bất đối xứng):**
   $$\text{Skewness} = \frac{1}{L} \sum_{i=1}^L \left(\frac{x_i - \mu}{\sigma}\right)^3$$
6. **Kurtosis (Độ nhọn):**
   $$\text{Kurtosis} = \frac{1}{L} \sum_{i=1}^L \left(\frac{x_i - \mu}{\sigma}\right)^4$$
7. **Crest Factor (Hệ số đỉnh):**
   $$C_f = \frac{\max(|x|)}{\text{RMS} + \epsilon}$$
8. **Shape Factor (Hệ số dạng):**
   $$S_f = \frac{\text{RMS}}{\frac{1}{L} \sum_{i=1}^L |x_i| + \epsilon}$$

*(Trong đó $\epsilon = 10^{-8}$ để đảm bảo an toàn số học).*

---

## 2. Ảnh hưởng của Điều kiện Vận hành (Operating Conditions)

Mặc dù các chỉ số thống kê trên không trực tiếp nhận tốc độ quay hay tải trọng làm tham số đầu vào trong công thức cơ bản, giá trị của chúng **phụ thuộc gián tiếp rất mạnh vào điều kiện vận hành**:
*   **Khi Tốc độ (Speed) / Tải trọng (Load) tăng:** Rung động cơ học của hệ thống tăng $\to$ Biên độ tín hiệu $x$ tăng vọt $\to$ Làm tăng các chỉ số biên độ tuyệt đối như **RMS, Peak-to-Peak, Std**.
*   **Cách khắc phục sự dịch chuyển phân phối:** Trong pipeline thực nghiệm, thuật toán **POT (Peak-Over-Threshold)** được áp dụng để tự động hiệu chuẩn ngưỡng động một cách độc lập cho từng vòng bi (tương ứng với từng ca vận hành riêng biệt) dựa trên tập điểm MSE lành mạnh (Healthy Scores) thu được ở giai đoạn đầu chu kỳ của vòng bi đó (khoảng 40% dữ liệu đầu theo cấu hình `train_ratio: 0.4`). Phương pháp hiệu chuẩn cục bộ (Local Calibration) này giúp triệt tiêu ảnh hưởng trôi biên độ do thay đổi điều kiện vận hành mà không cần nhúng trực tiếp vector điều kiện vận hành $oc$ vào mô hình HybridMamba. Vector điều kiện vận hành $oc$ được trích xuất trong tập dữ liệu thực tế chỉ dùng làm đầu vào so sánh cho mô hình baseline đối chứng (`MambaTSOfficial`).

---