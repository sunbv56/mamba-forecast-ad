# 🗑️ Báo Cáo Phân Tích & Phân Loại File Rác / Dư Thừa (Trash & Redundancy Analysis Report)
Báo cáo này liệt kê và phân loại toàn bộ các file trong dự án theo 5 mức độ từ rác hoàn toàn đến các file cốt lõi đang hoạt động, giúp bạn lên kế hoạch dọn dẹp dung lượng và tối ưu hóa cấu trúc dự án.

## 📊 Tóm Tắt Tổng Quan (Overview Summary)
| Mức độ | Số lượng file | Tổng dung lượng | Tỷ lệ dung lượng | Khuyến nghị hành động |
| :---: | :---: | :---: | :---: | :--- |
| Mức 5 | 28 | 4.06 KB | 0.00% | Rác hoàn toàn (0-byte, log thừa, patch, tmp) - **Xóa ngay** |
| Mức 4 | 71 | 55.16 MB | 0.13% | Trùng lặp & Dư thừa nặng (Thư mục copy lỗi, notebook cũ) - **Nên xóa** |
| Mức 3 | 49 | 11.27 GB | 27.99% | Tài liệu cũ / File nén backup / CV / Nhật ký thực tập - **Di chuyển hoặc Xóa** |
| Mức 2 | 821 | 53.75 MB | 0.13% | Tài liệu tham khảo / Thư viện hướng dẫn agent (Ít thay đổi) - **Giữ lại tham khảo** |
| Mức 1 | 86,861 | 28.88 GB | 71.75% | File cốt lõi (Code, Config, Data hoạt động, Notebook chính) - **BẮT BUỘC GIỮ** |
| **Tổng cộng** | **87,830** | **40.26 GB** | **100%** | **Tiết kiệm tiềm năng ~11.5 GB (~28.6% dung lượng)** |

---

## 🔍 Chi Tiết Phân Loại Từng Mức Độ (Detailed Level-by-Level Analysis)

### 🔴 Mức 5: Rác hoàn toàn (Completely Trash / Safe to Delete)
> [!IMPORTANT]
> Các file này hoàn toàn không có giá trị hoạt động (file trống 0-byte, file nháp tạm thời, file lỗi hệ thống). Bạn có thể xóa toàn bộ nhóm này ngay lập tức.

| STT | Đường dẫn file | Kích thước | Mô tả lý do |
| :---: | :--- | :---: | :--- |
| 1 | `tmp.md` | 2.38 KB | File nháp ghi chép tạm bảng so sánh kết quả |
| 2 | `fix_train.patch` | 1.18 KB | File Git patch dự phòng cũ |
| 3 | `output.txt` | 512 B | File log tham số chạy nháp |
| 4 | `rtk` | 0 Bytes | File trống không rõ nguồn gốc ở gốc dự án |
| 5 | `academic_research_skills/academic-pipeline/references/adapters/.gitkeep` | 0 Bytes | File giữ thư mục trống (nhiều thư mục đã có file hoặc không dùng) |
| 6 | `academic_research_skills/scripts/adapters/examples/folder_scan/input_fixture/Chen2024_AIAssessment.pdf` | 0 Bytes | File cấu hình trống hoặc log nháp |
| 7 | `academic_research_skills/scripts/adapters/examples/folder_scan/input_fixture/paper1.pdf` | 0 Bytes | File cấu hình trống hoặc log nháp |
| 8 | `academic_research_skills/scripts/adapters/examples/folder_scan/input_fixture/Wang_2023_formative_feedback.pdf` | 0 Bytes | File cấu hình trống hoặc log nháp |
| 9 | `academic_research_skills/scripts/adapters/examples/folder_scan/input_fixture/中文檔名_2024.pdf` | 0 Bytes | File cấu hình trống hoặc log nháp |
| 10 | `academic_research_skills/scripts/adapters/examples/obsidian/input_fixture/vault/.gitkeep` | 0 Bytes | File giữ thư mục trống (nhiều thư mục đã có file hoặc không dùng) |
| 11 | `academic_research_skills/scripts/adapters/examples/zotero/input_fixture/.gitkeep` | 0 Bytes | File giữ thư mục trống (nhiều thư mục đã có file hoặc không dùng) |
| 12 | `academic_research_skills/scripts/adapters/tests/.gitkeep` | 0 Bytes | File giữ thư mục trống (nhiều thư mục đã có file hoặc không dùng) |
| 13 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/raw/.gitkeep` | 0 Bytes | File giữ thư mục trống (nhiều thư mục đã có file hoặc không dùng) |
| 14 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/eval/.gitkeep` | 0 Bytes | File giữ thư mục trống (nhiều thư mục đã có file hoặc không dùng) |
| 15 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/models/.gitkeep` | 0 Bytes | File giữ thư mục trống (nhiều thư mục đã có file hoặc không dùng) |
| 16 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/.gitkeep` | 0 Bytes | File giữ thư mục trống (nhiều thư mục đã có file hoặc không dùng) |
| 17 | `scripts/run_pipeline.sh` | 0 Bytes | Shell script trống (0-byte) |
| 18 | `src/models/baselines/__init__.py` | 0 Bytes | File cấu hình trống hoặc log nháp |
| 19 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/processed/B01` | N/A (Inaccessible) | File/Thư mục bị lỗi hệ thống (WinError 1920, broken link) |
| 20 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/processed/B02` | N/A (Inaccessible) | File/Thư mục bị lỗi hệ thống (WinError 1920, broken link) |
| 21 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/processed/B03` | N/A (Inaccessible) | File/Thư mục bị lỗi hệ thống (WinError 1920, broken link) |
| 22 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/processed/B04` | N/A (Inaccessible) | File/Thư mục bị lỗi hệ thống (WinError 1920, broken link) |
| 23 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/processed/B05` | N/A (Inaccessible) | File/Thư mục bị lỗi hệ thống (WinError 1920, broken link) |
| 24 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/processed/B08` | N/A (Inaccessible) | File/Thư mục bị lỗi hệ thống (WinError 1920, broken link) |
| 25 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/processed/B10` | N/A (Inaccessible) | File/Thư mục bị lỗi hệ thống (WinError 1920, broken link) |
| 26 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/processed/B11` | N/A (Inaccessible) | File/Thư mục bị lỗi hệ thống (WinError 1920, broken link) |
| 27 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/processed/B12` | N/A (Inaccessible) | File/Thư mục bị lỗi hệ thống (WinError 1920, broken link) |
| 28 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/data/processed/B17` | N/A (Inaccessible) | File/Thư mục bị lỗi hệ thống (WinError 1920, broken link) |

### 🟠 Mức 4: Trùng lặp & Dư thừa nặng (Redundant / Duplicated)
> [!WARNING]
> Nhóm này chứa các bản sao lưu trùng lặp của mã nguồn, đặc biệt là thư mục viết sai chính tả `Hydrid-Mamba-for-Predictive-Bearing-Fault` (chứa clone của dự án chính) và các file Notebook cũ dung lượng lớn (9MB - 14MB). Xóa nhóm này sẽ giảm thiểu đáng kể sự nhiễu loạn thông tin.

| STT | Đường dẫn file | Kích thước | Mô tả lý do |
| :---: | :--- | :---: | :--- |
| 1 | `src/notebooks/tmp-test-final-mamba-forecast-ad.ipynb` | 13.97 MB | Notebook nháp tạm thời dung lượng rất lớn (14MB) |
| 2 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/test-final-mamba-forecast-ad.ipynb` | 12.08 MB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 3 | `notebooks/train-test-2-final-mamba-forecast-ad_old.ipynb` | 9.41 MB | Notebook phiên bản cũ đã được thay thế |
| 4 | `notebooks/train-test-final-mamba-forecast-ad.ipynb` | 9.40 MB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 5 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/models/mamba1_hybrid_default_best.pth` | 1.29 MB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 6 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_19_output_5_5.png` | 833.42 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 7 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_20_output_5_12.png` | 833.30 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 8 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_19_output_11_8.png` | 800.92 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 9 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_20_output_11_15.png` | 800.83 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 10 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_19_output_7_6.png` | 773.73 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 11 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_20_output_7_13.png` | 773.67 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 12 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_19_output_9_7.png` | 671.92 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 13 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_20_output_9_14.png` | 671.81 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 14 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_19_output_3_4.png` | 500.51 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 15 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_20_output_3_11.png` | 500.40 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 16 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_19_output_1_3.png` | 426.98 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 17 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_20_output_1_10.png` | 426.86 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 18 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_19_output_13_9.png` | 344.88 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 19 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_20_output_13_16.png` | 344.80 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 20 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_18_output_4_2.png` | 97.70 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 21 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_18_output_1_0.png` | 74.33 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 22 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/results/visualizations/cell_18_output_3_1.png` | 72.68 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 23 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/huong_dan_review_bai_bao_Q1.md` | 29.66 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 24 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/src/training/train.py` | 28.59 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| 25 | `Hydrid-Mamba-for-Predictive-Bearing-Fault/src/training/eval.py` | 23.55 KB | Nằm trong thư mục clone trùng lặp (Hydrid-Mamba...) |
| ... | *Và 46 file nhỏ khác trong thư mục `Hydrid-Mamba-for-Predictive-Bearing-Fault`* | | |

### 🟡 Mức 3: Tài liệu cũ / Lưu trữ cá nhân / File nén (Archived / Personal Drafts)
> [!NOTE]
> Chứa các file nén dữ liệu cũ khổng lồ (`_processed.zip` ~7.9GB, `processed.zip` ~2.7GB) và các tài liệu học tập/thực tập cá nhân của sinh viên (CV, Nhật ký thực tập, báo cáo định kỳ cũ). Nên di chuyển các file này ra ổ đĩa khác để lưu trữ nếu cần, hoặc xóa đi vì dữ liệu thực tế đã được giải nén sẵn trong thư mục `data/`.

| STT | Đường dẫn file | Kích thước | Mô tả lý do |
| :---: | :--- | :---: | :--- |
| 1 | `data/processed/_processed.zip` | 7.76 GB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 2 | `data/processed/processed.zip` | 2.69 GB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 3 | `data/raw/ieee-phm-2012-data-challenge-dataset-master.zip` | 728.06 MB | File nén dataset gốc từ IEEE PHM 2012 |
| 4 | `reports/11.05.2026-20260511T043139Z-3-001.zip` | 34.82 MB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 5 | `mamba-sft-source.zip` | 25.14 MB | Mã nguồn thư viện Mamba dạng zip dự phòng |
| 6 | `reports/11-05-2026-20260513T055210Z-3-001.zip` | 8.22 MB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 7 | `results/models/models.zip` | 6.99 MB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 8 | `reports/2-Tập Sự Nghề Nghiệp-20260524T055147Z-3-001/2-Tập Sự Nghề Nghiệp/BM01_XacNhanThucTap.docx` | 4.40 MB | Tài liệu thực tập cá nhân & CV học tập |
| 9 | `reports/2-Tập Sự Nghề Nghiệp-20260524T055147Z-3-001/2-Tập Sự Nghề Nghiệp/5_52100322_BaoCao_KTCN.docx` | 4.13 MB | Tài liệu thực tập cá nhân & CV học tập |
| 10 | `reports/2-Tập Sự Nghề Nghiệp-20260524T055147Z-3-001.zip` | 3.67 MB | Tài liệu thực tập cá nhân & CV học tập |
| 11 | `reports/report_2026_05_15.pdf` | 2.74 MB | Báo cáo tiến độ cũ theo ngày |
| 12 | `reports/2-Tập Sự Nghề Nghiệp-20260524T055147Z-3-001/2-Tập Sự Nghề Nghiệp/5_52100322_BaoCao.pdf` | 2.72 MB | Tài liệu thực tập cá nhân & CV học tập |
| 13 | `reports/11-05-2026-20260513T055210Z-3-001/11-05-2026/FEMamba_A_Feature-Enhanced_Mamba_Framework_with_Degradation-Stage_Global_Regularization_for_Bearing_Remaining_Useful_Life_Prediction.pdf` | 2.71 MB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 14 | `reports/11-05-2026-20260513T055210Z-3-001/11-05-2026/Chen_2024_Meas._Sci._Technol._35_106132.pdf` | 2.53 MB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 15 | `reports/report_2026_05_22.pdf` | 2.42 MB | Báo cáo tiến độ cũ theo ngày |
| 16 | `reports/2-Tập Sự Nghề Nghiệp-20260524T055147Z-3-001/2-Tập Sự Nghề Nghiệp/2_52100322_BM02.pdf` | 1.70 MB | Tài liệu thực tập cá nhân & CV học tập |
| 17 | `reports/report_2026_05_16.pdf` | 1.52 MB | Báo cáo tiến độ cũ theo ngày |
| 18 | `reports/11-05-2026-20260513T055210Z-3-001/11-05-2026/AI-Based_Application_for_Task_Management_and_Scheduling_Student_Activity.pdf` | 1.41 MB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 19 | `reports/11-05-2026-20260513T055210Z-3-001/11-05-2026/Bearing_Degradation_Prediction_based_on_Multi-Scale_Mamba-Transformer_Model.pdf` | 1.27 MB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 20 | `reports/11-05-2026-20260513T055210Z-3-001/11-05-2026/ConvMamba_A_Data-Efficient_Neural_Network_for_Bearing_Fault_Diagnosis.pdf` | 1.18 MB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 21 | `reports/2-Tập Sự Nghề Nghiệp-20260524T055147Z-3-001/2-Tập Sự Nghề Nghiệp/4_52100322_BM04.pdf` | 998.35 KB | Tài liệu thực tập cá nhân & CV học tập |
| 22 | `reports/report_2026_05_14.pdf` | 876.85 KB | Báo cáo tiến độ cũ theo ngày |
| 23 | `reports/KeHoach_Slide_ThuyetTrinh.png` | 668.77 KB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 24 | `TSP_template.zip` | 655.08 KB | File nén backup dữ liệu (Đã được giải nén trong project) |
| 25 | `reports/2-Tập Sự Nghề Nghiệp-20260524T055147Z-3-001/2-Tập Sự Nghề Nghiệp/1_52100322_BM01.pdf` | 555.35 KB | Tài liệu thực tập cá nhân & CV học tập |
| ... | *Và 24 file báo cáo/tài liệu cũ khác* | | |

### 🟢 Mức 2: Tài nguyên tham chiếu / Hướng dẫn (References & Guideline Materials)
> [!TIP]
> Gồm các tài liệu hướng dẫn viết bài báo Q1, hướng dẫn chạy code, các bài báo khoa học tham khảo (PDF trong `papers_ref`) và các thư viện guidelines cho AI agent (`academic_research_skills`, `karpathy_skills`). Chúng không tham gia chạy code trực tiếp nhưng rất có ích cho việc nghiên cứu và hỗ trợ viết tài liệu khoa học.

**Các thư mục / File chính trong nhóm này:**
1. `papers_ref/` (8 file PDFs bài báo nghiên cứu gốc - 35.8 MB)
2. `academic_research_skills/` (Thư viện chỉ dẫn nghiên cứu học thuật của AI agent - 0.7 MB)
3. `karpathy_skills/` (Các ví dụ & tiêu chuẩn lập trình của AI agent)
4. Các file markdown hướng dẫn trong root: `configuration_guide.md`, `running_guide.md`, `huong_dan_review_bai_bao_Q1.md`, `paper_optimal_alpha_plan.md`, `paper_preparation_checklist.md`, `pipeline_logic.md`, `project_structure.md`, `source_code_description.md`.

### 🔵 Mức 1: Hoạt động tốt / Cốt lõi (Core Code & Data)
Các file cốt lõi cấu thành nên ứng dụng bao gồm:
- Toàn bộ mã nguồn chạy mô hình trong `src/` (trừ notebook nháp).
- Các file cấu hình thiết lập thử nghiệm trong `configs/` (`default.yaml`, `nano.yaml`, `eval_pronostia_with_nano.yaml`).
- Toàn bộ dữ liệu thực nghiệm đã được xử lý đang dùng để train/test trong `data/processed/`, `data/splits/`.
- Các script chạy thí nghiệm tự động trong `scripts/` (`run_ablation_stats_pot.py`, `run_sensitivity_analysis.py`, `precompute_rms.py`).
- Các notebook đánh giá kết quả cuối cùng trong `notebooks/final/` và các file notebook chính.
- Các file quản lý môi trường và chạy dự án: `main.py`, `requirements.txt`, `README.md`.