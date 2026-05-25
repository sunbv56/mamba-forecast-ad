# HƯỚNG DẪN REVIEW BÀI BÁO THEO MỨC Q1

**Mục đích:**  
Tài liệu này dùng để tự kiểm tra, phản biện nội bộ hoặc hướng dẫn nhóm tác giả nâng một bản thảo từ mức hội thảo/Q4/Q3 lên định hướng **Q1 journal** trong các lĩnh vực như:

- Computer Science
- Artificial Intelligence
- Time-Series Analysis
- Industrial AI
- Prognostics and Health Management (PHM)
- Bearing Fault Diagnosis / Anomaly Detection
- IoT / AIoT / Smart Maintenance
- Software Engineering / Applied AI Systems

---

## 1. Khác biệt cốt lõi giữa Q4 và Q1

| Khía cạnh | Q4 thường chấp nhận | Q1 thường yêu cầu |
|---|---|---|
| Contribution | Incremental, ứng dụng được | Rõ ràng, có tính mới mạnh hoặc chứng minh sâu |
| Method | Có mô tả và triển khai được | Có rationale, formulation, ablation, theoretical/empirical justification |
| Dataset | 1 dataset có thể chấp nhận | Nên có nhiều dataset, nhiều setting hoặc benchmark chuẩn |
| Baseline | 2–3 baseline đơn giản/hợp lý | Baseline mạnh, SOTA, recent models, fair tuning |
| Evaluation | Metric cơ bản | Đầy đủ metric, robustness, ablation, sensitivity, statistical testing |
| Generalization | Có thể nói “potential” | Phải chứng minh qua cross-dataset, cross-condition, unseen setting |
| Writing | Rõ ràng, không overclaim | Lập luận học thuật chặt, gap sắc, contribution thuyết phục |
| Reproducibility | Mô tả đủ hiểu | Code/data/protocol rõ, seed, config, environment, complexity |
| Discussion | Nêu limitation | Phân tích sâu vì sao method hiệu quả/thất bại, implication, threat validity |

**Kết luận:**  
Q1 không chỉ cần “kết quả cao”. Q1 cần trả lời được câu hỏi:

> Bài này làm thay đổi hiểu biết, phương pháp, benchmark hoặc thực hành nghiên cứu trong lĩnh vực như thế nào?

---

# 2. Vai trò reviewer Q1

Bạn là **Associate Editor / Reviewer** cho một tạp chí Q1 trong lĩnh vực:

> [Điền lĩnh vực cụ thể: AI / PHM / Time-Series Anomaly Detection / Industrial IoT / Software Engineering / Education Technology /...]

Nhiệm vụ của bạn là đánh giá bài báo theo tiêu chuẩn Q1, tập trung vào:

1. **Novelty rõ ràng và có sức nặng**
2. **Research gap có bằng chứng từ literature gần đây**
3. **Method có cơ sở khoa học, không chỉ ghép module**
4. **Experimental design đủ mạnh để chứng minh claim**
5. **So sánh công bằng với baseline mạnh và SOTA**
6. **Ablation, sensitivity, robustness, generalization**
7. **Reproducibility và transparency**
8. **Discussion sâu, không quảng cáo kết quả**
9. **Ethics, data usage, citation integrity**
10. **Writing craft đủ thuyết phục reviewer khó tính**

---

# 3. Nguyên tắc bắt buộc khi review Q1

## 3.1. Nguyên tắc “Claim phải ngang bằng bằng chứng”

| Mức claim | Điều kiện tối thiểu |
|---|---|
| Mô tả hệ thống/phương pháp | Có kiến trúc, workflow, formulation |
| Khả thi | Có implementation và thử nghiệm cơ bản |
| Hiệu quả | Có baseline, metric, setup, repeated runs |
| Vượt trội | Có baseline mạnh, SOTA, fair tuning, statistical evidence |
| Robust | Có robustness test: noise, missing data, domain shift, unseen condition |
| Generalizable | Có nhiều dataset, nhiều domain/condition, cross-dataset validation |
| Real-world ready | Có pilot, deployment, latency/resource analysis, operational constraints |

**Nếu chưa có bằng chứng, không được dùng các claim sau:**

- state-of-the-art
- robust
- generalizable
- superior
- production-ready
- real-world ready
- comprehensive
- optimal
- solves the problem
- guarantees
- universally applicable

---

## 3.2. Q1 không chấp nhận “module stacking” nếu thiếu lý do

Với bài AI/ML, đặc biệt là các bài kiểu:

> “Chúng tôi kết hợp A + B + C + D để tạo framework mới.”

Reviewer Q1 sẽ hỏi:

1. Vì sao cần từng module?
2. Module đó giải quyết failure mode nào?
3. Có ablation chứng minh module đó cần thiết không?
4. Nếu bỏ module đó thì kết quả giảm thế nào?
5. Module đó có làm tăng complexity quá nhiều không?
6. Có phương án đơn giản hơn đạt kết quả tương đương không?
7. Có giải thích cơ chế hoạt động hay chỉ báo cáo điểm số?

**Với bài Mamba + CNN + RevIN + Decomposition + POT**, cần chứng minh:

| Module | Cần chứng minh |
|---|---|
| Mamba / SSM | Bắt long-range dependency tốt hơn TCN/LSTM/Transformer trong setting cụ thể |
| CNN branch | Cải thiện local transient / impulse detection |
| RevIN | Giảm distribution shift giữa operating conditions |
| Series decomposition | Tách trend/seasonal giúp anomaly score ổn định hơn |
| Multi-scale patching | Bắt được fault signature ở nhiều tần số/thang thời gian |
| POT / EVT threshold | Giảm false alarm tốt hơn fixed threshold, percentile threshold, z-score threshold |

---

# 4. Bộ tiêu chí review Q1 — Thang 100 điểm

## P1. Title, Abstract, Keywords — 8 điểm

### Checklist

- Tiêu đề nêu rõ problem, domain, method chính.
- Không dùng claim mạnh như “high precision”, “robust”, “SOTA” nếu chưa thật sự chứng minh.
- Abstract có đủ:
  - Bối cảnh
  - Gap
  - Method
  - Dataset
  - Baseline
  - Metric
  - Kết quả chính
  - Limitation hoặc scope
- Keywords gồm 4–8 cụm, đúng thuật ngữ chuyên ngành.

### Điểm mạnh cần tìm

- Tiêu đề specific, không quảng cáo.
- Abstract có số liệu định lượng rõ.
- Abstract không hứa quá mức.

### Vấn đề thường gặp

- Abstract chỉ mô tả method mà không có kết quả.
- Dùng “significantly” nhưng không có statistical test.
- Dùng “robust” nhưng chỉ test 1 dataset.

### Đề xuất cho bài Mamba-bearing

Không nên viết:

> High Precision Bearing Anomaly Detection Using HybridMamba++

Nên viết:

> Multi-Scale Series-Decomposed State Space Modeling for Bearing Anomaly Detection under Non-Stationary Vibration Signals

---

## P2. Introduction and Problem Significance — 12 điểm

### Checklist

- Bài toán quan trọng được chứng minh bằng citation/số liệu.
- Có bối cảnh công nghiệp/khoa học rõ.
- Nêu được limitation của nhóm phương pháp hiện tại.
- Research gap hẹp, sắc, có bằng chứng.
- Research questions hoặc hypotheses rõ.
- Contributions cụ thể và kiểm chứng được.

### Q1 yêu cầu thêm

Introduction phải làm tốt mô hình CARS:

1. **Establish territory:** lĩnh vực quan trọng thế nào?
2. **Establish niche:** existing work thiếu gì, fail ở đâu?
3. **Occupy niche:** bài này lấp gap bằng cách nào?

### Các câu hỏi reviewer Q1 sẽ đặt

- Gap này có thật không, hay chỉ là tác giả tự nói?
- Gap đã được chứng minh bằng Related Work chưa?
- Contribution có khác biệt thật với FEMamba, TFG-Mamba, TimesNet, ModernTCN, PatchTST, Anomaly Transformer không?
- Có phải chỉ ghép nhiều module đã có?
- Vì sao bài này đáng đăng ở Q1 thay vì Q3/Q4?

---

## P3. Related Work and Gap Positioning — 12 điểm

### Checklist

Related Work phải được phân nhóm theo logic nghiên cứu, không liệt kê từng bài rời rạc.

Gợi ý nhóm cho bài PHM / Bearing Anomaly Detection:

1. Traditional signal processing for bearing diagnosis
2. Deep learning for vibration-based fault detection
3. Time-series anomaly detection models
4. Transformer-based time-series models
5. State Space Models / Mamba for time series and PHM
6. Thresholding methods for anomaly detection
7. Distribution shift and domain adaptation in industrial signals

### Q1 yêu cầu

- Có tài liệu mới trong 3–5 năm gần nhất.
- Có bảng literature comparison.
- Có phân tích limitation trung thực.
- Không “đánh rơm” bài trước.
- Phải chỉ rõ bài hiện tại khác gì so với từng nhóm.

### Bảng literature matrix bắt buộc nên có

| Study | Task | Dataset | Method | Strength | Limitation | Relation to this work |
|---|---|---|---|---|---|---|

### Dấu hiệu chưa đạt Q1

- Related Work chỉ có 8–12 tài liệu.
- Thiếu bài SOTA gần đây.
- Không có bảng so sánh.
- Không giải thích rõ gap.
- Citation không thật sự hỗ trợ claim.

---

## P4. Methodology and Technical Soundness — 16 điểm

### Checklist

Methodology phải đủ chi tiết để reviewer hiểu và tái triển khai ở mức hợp lý.

Cần có:

1. Problem formulation
2. Input/output definition
3. Dataset notation
4. Preprocessing pipeline
5. Model architecture
6. Training objective
7. Anomaly scoring
8. Thresholding mechanism
9. Complexity analysis
10. Implementation details

---

## 4.1. Problem formulation bắt buộc

Ví dụ với bearing anomaly detection:

- Input: vibration sequence \(X \in \mathbb{R}^{T \times C}\)
- Window: \(x_i \in \mathbb{R}^{L \times C}\)
- Reconstruction: \(\hat{x}_i = f_\theta(x_i)\)
- Anomaly score: \(s_i = \|x_i - \hat{x}_i\|_2^2\)
- Threshold: \(\tau\) estimated by POT/EVT
- Decision: \(y_i = 1\) if \(s_i > \tau\), otherwise \(0\)

### Reviewer sẽ hỏi

- Window length chọn thế nào?
- Sampling rate bao nhiêu?
- Có overlap không?
- Có leakage giữa train/test không?
- Train chỉ dùng healthy hay dùng cả fault?
- Anomaly label xác định theo timestamp hay condition label?
- Threshold fit trên validation healthy hay mixed data?
- Có dùng thông tin tương lai gây leakage không?

---

## 4.2. Kiến trúc model cần trình bày rõ

Với bài HybridMamba++ nên có sơ đồ:

```text
Raw vibration signal
        |
Windowing / Normalization
        |
RevIN
        |
Multi-scale decomposition
        |
+----------------------+----------------------+
| CNN local branch     | Mamba global branch  |
| transient features   | long-range dynamics  |
+----------------------+----------------------+
        |
Feature fusion
        |
Decoder / Reconstruction head
        |
MSE anomaly score
        |
POT threshold
        |
Normal / Anomaly decision
```

### Cần mô tả cụ thể

- CNN kernel size
- Number of Mamba layers
- Hidden dimension
- State dimension
- Fusion strategy
- Decoder type
- Loss function
- Optimizer
- Learning rate
- Batch size
- Epochs
- Early stopping
- Random seed
- Parameter count
- FLOPs hoặc inference latency

---

## P5. Experimental Design — 16 điểm

### Checklist Q1

- Dataset rõ ràng, nguồn hợp pháp.
- Train/validation/test split chống leakage.
- Baseline đủ mạnh.
- Metric phù hợp.
- Có repeated runs.
- Có standard deviation hoặc confidence interval.
- Có statistical significance nếu claim mạnh.
- Có ablation.
- Có robustness/sensitivity/generalization test.
- Có fair tuning cho baseline.

---

## 5.1. Dataset requirement

Với bài bearing/PHM, nên có tối thiểu:

### Mức tối thiểu cho Q1 yếu

- 1 dataset chính
- split nghiêm ngặt
- nhiều baseline mạnh
- ablation đầy đủ
- robustness test

### Mức tốt hơn

- 2–3 dataset:
  - CWRU
  - Paderborn
  - XJTU-SY
  - IMS
  - PRONOSTIA
  - Mendeley run-to-failure bearing dataset
  - PU bearing dataset

### Mức rất mạnh

- Cross-dataset validation:
  - train trên dataset A, test trên dataset B
  - train trên operating condition A, test trên condition B
  - test unseen fault severity / unseen load / unseen speed

---

## 5.2. Baseline requirement

Nếu claim tốt hơn SOTA, phải so với:

### Classical / ML baselines

- PCA / One-Class SVM
- Isolation Forest
- Autoencoder
- LSTM-AE
- CNN-AE

### Time-series deep baselines

- TCN
- ModernTCN
- TimesNet
- PatchTST
- DLinear / NLinear
- Informer / Autoformer / FEDformer nếu phù hợp

### Anomaly detection baselines

- Anomaly Transformer
- TranAD
- USAD
- OmniAnomaly
- THOC
- MSCRED nếu phù hợp

### Mamba/SSM baselines

- Mamba
- S-Mamba / TimeMachine / Mamba-based time-series model nếu liên quan
- FEMamba / TFG-Mamba nếu cùng PHM/RUL/fault diagnosis

### Threshold baselines

- Fixed threshold
- Percentile threshold
- z-score threshold
- IQR threshold
- Gaussian threshold
- POT/EVT

---

## 5.3. Metrics requirement

Với anomaly detection:

| Metric | Ý nghĩa |
|---|---|
| Precision | Cảnh báo đúng trong số cảnh báo |
| Recall | Bắt được bao nhiêu anomaly |
| F1-score | Cân bằng precision/recall |
| AUROC | Phân biệt normal/anomaly |
| AUPRC | Quan trọng khi anomaly hiếm |
| FAR | False Alarm Rate |
| FNR | Missed anomaly rate |
| Detection Delay | Phát hiện sớm hay trễ |
| Latency | Thời gian inference |
| Parameter count | Độ nhẹ model |
| FLOPs | Chi phí tính toán |
| Memory usage | Khả năng chạy edge |

Với PHM thực tế, nên thêm:

- detection lead time
- early warning time
- false alarms per hour/day
- inference throughput
- resource footprint trên edge device nếu claim deployment

---

## P6. Ablation Study — 10 điểm

Q1 gần như bắt buộc phải có ablation nếu bài claim nhiều contribution.

### Ablation tối thiểu cho HybridMamba++

| Variant | Mục tiêu kiểm tra |
|---|---|
| Full model | Kết quả chính |
| w/o RevIN | RevIN có giảm distribution shift không? |
| w/o decomposition | Decomposition có giúp anomaly score ổn định không? |
| w/o CNN branch | CNN có giúp local transient không? |
| w/o Mamba branch | Mamba có giúp long-range dependency không? |
| single-scale patching | Multi-scale có cần thiết không? |
| fixed threshold instead of POT | POT có giảm false alarm không? |
| Transformer instead of Mamba | SSM có lợi thế complexity/performance không? |
| TCN instead of Mamba | So với receptive field cố định |
| no fusion / late fusion / early fusion | Fusion strategy có quan trọng không? |

### Reviewer sẽ hỏi

- Mỗi module đóng góp bao nhiêu?
- Module nào quan trọng nhất?
- Có module nào không cần thiết không?
- Model có phức tạp quá mức so với lợi ích không?

---

## P7. Robustness, Generalization, and Stress Testing — 10 điểm

### Q1 yêu cầu thêm

Nếu bài dùng từ “robust” hoặc hướng tới công nghiệp, cần test:

1. Noise injection
2. Missing sensor data
3. Sensor drift
4. Different load/speed conditions
5. Unseen fault type
6. Unseen fault severity
7. Cross-machine validation
8. Domain shift
9. Small training data regime
10. Threshold calibration sensitivity

### Bảng stress test nên có

| Scenario | Description | Metric drop | Interpretation |
|---|---|---:|---|
| Gaussian noise | SNR = 20/10/5 dB | ... | ... |
| Missing segment | 5%/10%/20% missing | ... | ... |
| Load shift | train load A, test load B | ... | ... |
| Speed shift | train speed A, test speed B | ... | ... |
| Fault severity shift | unseen severity | ... | ... |

### Không có robustness test thì không được claim

- robust
- reliable under industrial noise
- generalizable
- real-world ready

---

## P8. Results Presentation and Statistical Evidence — 8 điểm

### Checklist

- Bảng kết quả chính rõ.
- Figure có caption tự hiểu được.
- Có mean ± std nếu chạy nhiều seed.
- Có ranking giữa model.
- Có bold/underline nhưng không lạm dụng.
- Có test ý nghĩa thống kê nếu claim “significant”.
- Không cherry-pick.

### Statistical evidence nên có

- Run 3–5 random seeds
- Report mean ± std
- Paired t-test hoặc Wilcoxon signed-rank test nếu phù hợp
- Confidence interval cho metric chính
- Effect size nếu muốn mạnh hơn

### Câu claim phù hợp

Không nên viết:

> The proposed model significantly outperforms all baselines.

Nên viết:

> The proposed model achieved the highest AUPRC among the evaluated baselines, with an average improvement of X% over the strongest baseline across five runs.

---

## P9. Discussion, Threats to Validity, and Limitations — 8 điểm

### Q1 yêu cầu Discussion sâu

Discussion không chỉ nhắc lại bảng điểm.

Cần trả lời:

1. Vì sao model tốt hơn?
2. Khi nào model thất bại?
3. Module nào giúp nhiều nhất?
4. Có trade-off accuracy vs latency không?
5. Threshold POT có nhạy với calibration set không?
6. Kết quả có phụ thuộc dataset không?
7. Có khả năng áp dụng công nghiệp đến mức nào?
8. Bài học rút ra cho cộng đồng nghiên cứu là gì?

---

## 9.1. Threats to validity nên chia nhóm

### Internal validity

- Data leakage
- Hyperparameter tuning bias
- Random seed variability
- Threshold calibration bias

### External validity

- Chỉ test một dataset
- Chưa test nhiều machine/load/speed
- Chưa có pilot công nghiệp

### Construct validity

- Label anomaly có phản ánh fault thật không?
- MSE reconstruction có phải proxy tốt cho anomaly không?
- FAR/AUPRC có đủ cho maintenance decision không?

### Conclusion validity

- Số lần chạy có đủ không?
- Có statistical test không?
- Có quá diễn giải kết quả không?

---

## P10. Reproducibility and Open Science — 6 điểm

### Checklist

- Có mô tả environment:
  - Python version
  - PyTorch/TensorFlow version
  - CUDA version
  - GPU/CPU/RAM
- Có seed.
- Có config.
- Có preprocessing detail.
- Có link code nếu journal cho phép.
- Có data access instruction.
- Có model hyperparameters.
- Có evaluation script hoặc protocol.

### Q1 rất ưu tiên

- GitHub repository
- requirements.txt / environment.yml
- config YAML
- random seed script
- data split file
- trained model checkpoint nếu được
- supplementary material

---

## P11. References and Scholarly Positioning — 6 điểm

### Checklist

- Reference thật, đúng format.
- Bao phủ seminal + recent works.
- Có tài liệu 3–5 năm gần nhất.
- Có bài từ journal/conference uy tín.
- Không dùng citation ảo.
- Không phụ thuộc quá nhiều vào arXiv nếu có bản peer-reviewed.
- Citation phải hỗ trợ đúng claim.

### Q1 yêu cầu

- Related Work không chỉ “nhiều reference”.
- Phải có synthesis.
- Phải chỉ ra được vị trí của bài trong bản đồ nghiên cứu.

---

## P12. Ethics, Data, and AI Disclosure — Pass/Fail

### Fail nếu có

- Fabricated results
- Citation không tồn tại
- Dùng AI tạo số liệu giả
- Đạo văn
- Self-plagiarism nghiêm trọng
- Không khai báo dữ liệu nhạy cảm nếu có
- Không tuân thủ license dataset
- Không tuân thủ policy dùng AI của journal

### Với bài dùng AI hỗ trợ viết/code

Cần kiểm tra policy tạp chí. Nếu cần, khai báo:

> The authors used AI-assisted tools for language polishing and code debugging. All scientific content, experimental design, analysis, and conclusions were verified by the authors.

---

# 5. Writing Craft cho Q1 — 20 điểm bổ sung

Điểm này không cộng trực tiếp vào 100 kỹ thuật, nhưng ảnh hưởng mạnh tới quyết định review.

## R1. Novelty Framing — 4 điểm

Bài phải trả lời rõ:

- Cái mới là gì?
- Mới so với ai?
- Mới ở mức method, task, dataset, protocol hay application?
- Vì sao cái mới đó quan trọng?

## R2. Argumentation Quality — 4 điểm

- Mỗi section có logic dẫn dắt.
- Claim nào cũng có evidence.
- Không nhảy từ problem sang solution quá nhanh.
- Không để reviewer phải tự suy luận contribution.

## R3. Citation Rhetoric — 3 điểm

- Citation theo cụm chủ đề.
- Có synthesis sau mỗi nhóm.
- Không liệt kê “A did X, B did Y, C did Z” đơn thuần.

## R4. Hedging and Precision — 3 điểm

Dùng ngôn ngữ chính xác:

- “under the evaluated setting”
- “on the selected datasets”
- “compared with the selected baselines”
- “the results suggest”
- “provides evidence”
- “indicates potential”

## R5. Technical Clarity — 3 điểm

- Notation thống nhất.
- Hình kiến trúc rõ.
- Pseudocode nếu cần.
- Equation không dư nhưng đủ.

## R6. Reviewer Persuasion — 3 điểm

Reviewer phải thấy:

- Bài đáng đọc.
- Gap đáng giải quyết.
- Method hợp lý.
- Results đáng tin.
- Limitation trung thực.
- Contribution đủ để đăng Q1.

---

# 6. Checklist Q1 riêng cho bài Bearing Anomaly Detection / PHM

## 6.1. Dataset

- [ ] Có mô tả dataset rõ: sensor, sampling rate, machine, load, speed, fault type.
- [ ] Có chia train/val/test chống leakage.
- [ ] Có giải thích healthy/anomaly label.
- [ ] Có nhiều condition hoặc nhiều dataset nếu claim generalization.
- [ ] Có phân tích class imbalance.

## 6.2. Signal preprocessing

- [ ] Window length và overlap rõ.
- [ ] Normalization rõ.
- [ ] FFT/STFT/wavelet nếu dùng phải mô tả.
- [ ] Không dùng thông tin test để normalize train.
- [ ] Không leakage qua overlapping windows giữa train/test.

## 6.3. Model

- [ ] Có sơ đồ architecture.
- [ ] Có formulation.
- [ ] Có hyperparameters.
- [ ] Có parameter count.
- [ ] Có latency/FLOPs nếu claim efficiency.

## 6.4. Evaluation

- [ ] Có baseline truyền thống.
- [ ] Có baseline deep learning.
- [ ] Có baseline time-series/SOTA.
- [ ] Có threshold baseline.
- [ ] Có ablation.
- [ ] Có robustness test.
- [ ] Có repeated runs.
- [ ] Có mean ± std.
- [ ] Có error analysis.

## 6.5. Practical relevance

- [ ] Có false alarms per time unit.
- [ ] Có detection delay/lead time.
- [ ] Có discussion về deployment.
- [ ] Có resource analysis.
- [ ] Có limitation về industrial pilot.

---

# 7. Mẫu bảng chấm điểm Q1

| Nhóm đánh giá | Điểm tối đa | Điểm đạt |
|---|---:|---:|
| P1. Title, Abstract, Keywords | 8 | |
| P2. Introduction and Problem Significance | 12 | |
| P3. Related Work and Gap Positioning | 12 | |
| P4. Methodology and Technical Soundness | 16 | |
| P5. Experimental Design | 16 | |
| P6. Ablation Study | 10 | |
| P7. Robustness and Generalization | 10 | |
| P8. Results and Statistical Evidence | 8 | |
| P9. Discussion and Limitations | 8 | |
| P10. Reproducibility | 6 | |
| P11. References | 6 | |
| **Tổng** | **112** | |

Có thể quy đổi về 100 bằng công thức:

```text
Final Score = Raw Score / 112 * 100
```

---

# 8. Ngưỡng quyết định theo chuẩn Q1

| Điểm quy đổi | Đánh giá |
|---:|---|
| ≥ 85 | Có tiềm năng Q1, cần minor/major revision tùy journal |
| 78–84 | Có hướng Q1 nhưng cần bổ sung thực nghiệm/viết lại đáng kể |
| 70–77 | Mức Q2/Q3 mạnh, chưa đủ Q1 |
| 60–69 | Có hướng nhưng thiếu novelty/evaluation |
| < 60 | Chưa đủ chuẩn journal tốt |

## Gợi ý quyết định reviewer

| Tình trạng | Quyết định |
|---|---|
| Novelty mạnh, experiments tốt, viết rõ | Minor Revision |
| Method tốt nhưng thiếu robustness/statistics | Major Revision |
| Ý tưởng tốt nhưng experiments yếu | Major Revision / Reject & Resubmit |
| Chỉ ghép module, thiếu novelty | Reject |
| Có lỗi đạo đức/data/citation | Reject |

---

# 9. Mẫu review Q1 cuối cùng

## TỔNG ĐIỂM

- Technical Score: `[x]/100`
- Writing Craft: `[y]/20`
- Ethics/Data/AI Disclosure: `Pass / Fail`
- Reproducibility Risk: `Low / Medium / High`
- Overclaim Risk: `Low / Medium / High`
- Q1 Readiness: `Low / Medium / High`

---

## KẾT LUẬN REVIEW

Bản thảo hiện tại `[phù hợp / chưa phù hợp]` với định hướng Q1 vì:

1. `[Lý do về novelty]`
2. `[Lý do về method]`
3. `[Lý do về experimental evidence]`
4. `[Lý do về writing/reproducibility]`

---

## ĐIỂM MẠNH CHÍNH

1. ...
2. ...
3. ...

---

## VẤN ĐỀ NGHIÊM TRỌNG CẦN SỬA

1. ...
2. ...
3. ...

---

## CÁC CLAIM CẦN GIẢM MỨC

| Claim hiện tại | Vấn đề | Câu nên sửa |
|---|---|---|
| ... | ... | ... |

---

## EXPERIMENTS CẦN BỔ SUNG ĐỂ HƯỚNG Q1

1. Cross-dataset validation
2. Strong SOTA baselines
3. Ablation từng module
4. Robustness under noise/missing data/domain shift
5. Statistical test over multiple runs
6. Complexity and latency analysis
7. Error analysis and failure cases

---

## KHUYẾN NGHỊ

- `Accept`
- `Minor Revision`
- `Major Revision`
- `Reject & Resubmit`
- `Reject`

---

# 10. Áp dụng nhanh cho bài HybridMamba++ / Bearing Anomaly Detection

## 10.1. Điểm hiện tại nếu theo chuẩn Q1

Nếu bản thảo chỉ có Abstract + Introduction và còn template ở các phần sau, thì chưa thể đánh giá Q1 đầy đủ. Mức hiện tại nên xem là:

```text
Q1 Readiness: Very Low
Suggested Decision: Reject & Resubmit internally
```

Lý do:

1. Related Work chưa hoàn chỉnh.
2. Methodology chưa đủ tái triển khai.
3. Results chưa có bảng/figure tương ứng.
4. References chưa thật.
5. Claim performance chưa có evidence trong bản thảo.
6. Chưa có ablation.
7. Chưa có robustness/generalization.
8. Chưa có reproducibility package.

---

## 10.2. Để nâng bài này lên hướng Q1, cần bổ sung tối thiểu

### A. Related Work mạnh

Cần viết đầy đủ các nhóm:

- Bearing fault diagnosis
- Time-series anomaly detection
- Transformer-based time-series models
- Mamba/SSM for time-series/PHM
- Decomposition-based forecasting/anomaly detection
- Thresholding and EVT/POT for anomaly detection
- Domain shift in industrial vibration signals

### B. Methodology đầy đủ

Cần có:

- Problem formulation
- Architecture diagram
- Mathematical description
- Training objective
- Anomaly score
- Threshold algorithm
- Complexity analysis
- Pseudocode

### C. Experimental setup mạnh

Cần có:

- Ít nhất 2 dataset nếu có thể
- Train/test split chống leakage
- Baseline mạnh
- Metric đầy đủ
- Repeated runs
- Hardware/software setup

### D. Ablation bắt buộc

- Full model
- w/o RevIN
- w/o decomposition
- w/o CNN
- w/o Mamba
- w/o POT
- single-scale vs multi-scale
- fixed threshold vs POT

### E. Robustness/generalization

- Noise test
- Missing data test
- Load/speed shift
- Cross-dataset test nếu có
- Threshold sensitivity

### F. Error analysis

Cần phân tích:

- False positives xảy ra ở đoạn nào?
- False negatives thuộc fault type nào?
- Anomaly score có tăng trước fault không?
- Detection delay bao nhiêu?
- Trường hợp nào POT threshold thất bại?

---

# 11. Mẫu bảng experiments nên có trong bài Q1

## Table 1. Dataset summary

| Dataset | Sensor | Sampling rate | Fault types | Conditions | Train/Val/Test |
|---|---|---:|---|---|---|

## Table 2. Main comparison

| Model | Precision | Recall | F1 | AUROC | AUPRC | FAR | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|

## Table 3. Ablation study

| Variant | F1 | AUPRC | FAR | Latency | Interpretation |
|---|---:|---:|---:|---:|---|

## Table 4. Robustness test

| Scenario | Model | AUPRC | F1 | FAR | Drop |
|---|---|---:|---:|---:|---:|

## Table 5. Complexity comparison

| Model | Params | FLOPs | Inference time | Memory |
|---|---:|---:|---:|---:|

## Table 6. Statistical significance

| Comparison | Metric | p-value | Effect size | Significant? |
|---|---|---:|---:|---|

---

# 12. Câu viết nên dùng khi hướng Q1

## Khi có kết quả nhưng chưa quá mạnh

> The results provide evidence that the proposed architecture can improve anomaly detection performance under the evaluated non-stationary vibration settings.

## Khi chỉ test 1 dataset

> While the results are promising on the evaluated dataset, further validation across additional machines and operating conditions is required before claiming general applicability.

## Khi có baseline nhưng chưa đủ SOTA

> The comparison focuses on representative baselines rather than an exhaustive state-of-the-art benchmark.

## Khi có ablation

> The ablation results indicate that the decomposition and threshold calibration components contribute most strongly to reducing false alarms.

## Khi chưa deploy thực tế

> The current study evaluates offline detection performance; real-time deployment and maintenance decision integration remain future work.

---

# 13. Checklist cuối trước khi gửi Q1

## Novelty

- [ ] Gap có thật và có citation hỗ trợ.
- [ ] Contribution không chỉ là ghép module.
- [ ] Có phân biệt rõ với SOTA.
- [ ] Có novelty về method/protocol/dataset/application.

## Method

- [ ] Có formulation.
- [ ] Có architecture.
- [ ] Có pseudocode.
- [ ] Có hyperparameters.
- [ ] Có complexity.

## Experiments

- [ ] Có strong baselines.
- [ ] Có fair tuning.
- [ ] Có repeated runs.
- [ ] Có mean ± std.
- [ ] Có ablation.
- [ ] Có robustness.
- [ ] Có statistical evidence.
- [ ] Có error analysis.

## Writing

- [ ] Không overclaim.
- [ ] Không văn phong quảng cáo.
- [ ] Không citation rỗng.
- [ ] Discussion có phân tích WHY.
- [ ] Limitation trung thực.

## Reproducibility

- [ ] Có seed.
- [ ] Có environment.
- [ ] Có config.
- [ ] Có data split protocol.
- [ ] Có code hoặc mô tả đủ rõ.

---

# 14. Kết luận

Để đạt mức Q1, một bài không thể chỉ dừng ở:

> “Tôi đề xuất một mô hình mới và kết quả cao.”

Mà phải chứng minh được:

1. **Gap là thật**
2. **Method giải quyết đúng gap**
3. **Từng module đều có lý do tồn tại**
4. **Kết quả vượt baseline mạnh một cách công bằng**
5. **Hiệu quả ổn định qua nhiều setting**
6. **Claim không vượt quá bằng chứng**
7. **Bài có giá trị cho cộng đồng nghiên cứu**

Với bài **HybridMamba++ cho bearing anomaly detection**, hướng Q1 khả thi hơn nếu tác giả biến bài từ một bản “model proposal” thành một nghiên cứu đầy đủ về:

> **Non-stationary, multi-scale, threshold-calibrated time-series anomaly detection for industrial vibration signals, validated through strong baselines, ablation, robustness, and cross-condition evaluation.**
