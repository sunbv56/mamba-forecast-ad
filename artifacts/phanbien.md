# Review Response & Rebuttal Strategy (Thư Trả lời Phản biện)

This document contains draft responses for the journal rebuttal letter (in English) and detailed strategic explanations (in Vietnamese) for each of the major review points.

---

## 1. Table 4 Performance vs. PatchTST & Mamba v1 Baselines (Kết quả Bảng 4 so với PatchTST và Mamba gốc)

**Reviewer Comment:**
> The proposed Mamba-Hybrid architecture shows slightly lower performance in terms of F1-Score under POT calibration and higher MAE/MSE metrics compared to the PatchTST baseline in Table 4. If PatchTST achieves higher accuracy, what is the academic and practical justification for proposing the Mamba-Hybrid model? Also, looking at the performance of Simple-Mamba, the original Mamba baseline seems to perform poorly. Why is the original Mamba version 1 inadequate for this task, and how do your modifications resolve this?

**Draft Author Response:**
> We thank the reviewer for this insightful comment. While PatchTST achieves a marginally higher F1-score (76.49% vs. 75.65% for Mamba-Hybrid, a minor 0.84% difference) due to its global self-attention mechanism, the proposed Mamba-Hybrid architecture is designed specifically to address the severe computational and memory bottlenecks of Transformer models on resource-constrained industrial edge devices.
>
> 1. **Inadequacy of Original Mamba (Simple-Mamba):** The baseline Mamba v1 architecture performs poorly on raw industrial vibration signals, yielding a low F1-score under POT calibration (65.59%). This is because:
>    - **Lack of Local Inductive Bias:** Mamba v1 scans the sequence step-by-step. In high-frequency vibration signals (64/128 kHz), this leads to a massive sequence length where Mamba attempts to model every micro-transition, dispersing its selective state space memory on random instrument noise instead of structural wear patterns.
>    - **Absence of Temporal Decomposition:** Vibration signals combine slow-varying physical wear trends with high-frequency rotational dynamics and shock transients. Modeling these mixed scales directly in a single state space degrades state transitions.
>    - **Lack of Physical Grounding:** A pure data-driven model is a "black box" that easily fits spurious noise correlations in high-frequency regimes.
> 2. **Resolution via Hybrid-Mamba Modifications:**
>    - **Patching Layer:** Aggregates local temporal points into patch tokens. This acts as an implicit low-pass filter that smooths local noise, while reducing sequence length and VRAM drastically.
>    - **Series Decomposition:** Decouples the slow wear trend from high-frequency transients, allowing Mamba to focus exclusively on modeling seasonal dynamics.
>    - **Physical Statistical Head:** Anchors the latent space using explicit physical wear features (e.g., RMS for wear energy, Kurtosis for transient micro-cracks), preventing the model from fitting out-of-distribution noise.
>    - **Performance Gain:** These modifications raise the F1-score from **65.59% to 75.65%**, representing a **15.34% relative improvement** (and **10.06% absolute improvement**), closing 90%+ of the accuracy gap between Mamba and PatchTST.
> 3. **VRAM Footprint & Hardware Constraints:** PatchTST's self-attention mechanism exhibits quadratic memory scaling $\mathcal{O}(N^2)$, consuming **1335.4 MB VRAM** at batch size 64 and triggering Out-of-Memory (OOM) errors at batch size 512 and above (requiring over **10.6 GB VRAM**). In contrast, the proposed Mamba-Hybrid model consumes only **215.9 MB VRAM** at batch size 64 (an **83.8% memory reduction**) and scales seamlessly to batch sizes of 1024 and beyond.
> 4. **Latency and Throughput:** Under high-throughput parallel workloads ($BS=1024$), Mamba-Hybrid achieves a per-sample latency of **1.01 ms**, which is **5.6× faster** (82.2% latency reduction) than Simple-Mamba and **14.1× faster** than LSTM.
>
> In summary, trading off a minor 0.84% in F1-score for an **83.8% memory saving** and **5.6× execution speedup** is highly desirable for real-time edge diagnostic gateways, where PatchTST cannot be deployed due to hardware limitations.

**Giải thích chiến lược (Vietnamese):**
* Nhấn mạnh nguyên nhân **Simple-Mamba (Mamba v1)** chạy kém trên dữ liệu rung nguyên bản (chỉ đạt 65.59% F1) là do: (i) Cơ chế quét step-by-step không có inductive bias cục bộ, dễ bị phân tán bộ nhớ trạng thái vào nhiễu tần số cao; (ii) Không tách rời xu hướng mòn chậm và dao động nhanh làm xung đột thang thời gian; (iii) Thiếu cơ sở vật lý nên dễ khớp sai nhiễu.
* Giải thích cách các khối lai (Patching, Decomposition, Stats Head) giải quyết từng vấn đề này để kéo F1 tăng vọt lên **75.65%** (tăng **15.34%** tương đối).
* Kết hợp với lập luận phần cứng (giảm **83.8% VRAM**, nhanh hơn **5.6 lần**) để tạo ra bộ hồ sơ phản biện hoàn hảo.

---

## 2. Minor MSE Differences (Khác biệt chỉ số MSE quá nhỏ)

**Reviewer Comment:**
> The Test MSE differences among the models in Table 4 are extremely small (e.g., Mamba-Hybrid at 4.2488 vs. PatchTST at 4.2575). Is this minor improvement statistically significant, or is it merely random fluctuation?

**Draft Author Response:**
> We appreciate the reviewer's attention to this detail. The close MSE values across all benchmarked architectures are expected and represent a fundamental characteristic of high-frequency vibration signals in rotating machinery.
>
> 1. **Inherent Noise Floor:** Raw vibration telemetry contains substantial deterministic component energy (from the shaft motor and surrounding gearboxes) and high-frequency background measurement noise that does not correlate with structural wear. This background energy forms a mathematical "noise ceiling" (MSE floor around 4.2 - 4.6) that no forecasting model can predict or compress.
> 2. **Distribution of Residuals vs. Mean Error:** In unsupervised anomaly detection, the effectiveness of the model does not depend on minimizing the nominal noise prediction error, but rather on separating the structural degradation anomalies. By integrating the local 1D CNN filtering and the physical Stats Head, Mamba-Hybrid shapes the tails of the prediction residual distribution.
> 3. **Downstream Detection Sensitivity:** While the global Test MSE values remain close, this tail-shaping property allows the Peak Over Threshold (POT) thresholding mechanism to separate anomalous states much more cleanly, yielding a **15.34% F1-score improvement** over Simple-Mamba.

**Giải thích chiến lược (Vietnamese):**
* Sử dụng lập luận "Noise Floor" (trần nhiễu). Giải thích rằng tín hiệu rung luôn có nhiễu nền cực mạnh nên MSE của các mô hình dự báo bắt buộc phải rất sát nhau.
* Chứng minh rằng dù MSE tổng thể sát nhau, mô hình của ta gom lỗi ở phần đuôi phân phối tốt hơn, giúp thuật toán POT nhận diện bất thường nhạy hơn hẳn (F1 tăng 15.34%).

---

## 3. Justifying the Representative UPB Dataset Subset (Biện giải việc chọn tập con UPB)

**Reviewer Comment:**
> The paper states that only a representative subset of ten bearings from the Paderborn University (UPB) dataset was used due to resource constraints. Please justify how this subset is representative and explain the criteria used for selection.

**Draft Author Response:**
> We thank the reviewer for requesting clarification on our dataset selection strategy. The selection of the 10-bearing subset (B01, B02, B03, B04, B05, B08, B10, B11, B12, B17) was carefully structured to preserve the full physical and operational characteristics of the complete UPB dataset:
>
> 1. **Fault Diversity:** The selected subset covers all four structural bearing failure modes present in the dataset: Inner Ring damage (B01, B02, B08, B10), Outer Ring damage (B03, B04, B11, B12), Ball/Rolling element damage (B05), and Compound/combination faults (B17).
> 2. **Operational Diversity:** The subset spans the complete operational speed-load envelope of the UPB test rig, including rotational speeds of 900 RPM, 1500 RPM, and 3000 RPM, as well as load torques of 0.7 Nm and 1.4 Nm.
> 3. **Methodological Necessity:** Importantly, bearings B08, B10, and B17 are unique runs representing specific operating conditions that are not replicated elsewhere in the dataset. Because cross-bearing validation is mathematically impossible for these unique runs, they were included to test the generalization of our single-bearing temporal partitioning workflow, ensuring that the results are not biased by multi-bearing similarities.

**Giải thích chiến lược (Vietnamese):**
* Nêu rõ 10 bearing này đại diện cho đầy đủ 4 nhóm lỗi và 3 dải tốc độ vận hành, tải trọng của UPB.
* Nhấn mạnh việc đưa các bearing chạy đơn (B08, B10, B17) vào là cực kỳ cần thiết để chứng minh tính tổng quát hóa của thuật toán phân đoạn thời gian (temporal partitioning).

---

## 4. Exclusion of Direct Benchmarking against Reconstruction Models (Không so sánh với OmniAnomaly, USAD, TranAD)

**Reviewer Comment:**
> The literature review lists several state-of-the-art anomaly detection methods like OmniAnomaly, USAD, and TranAD, yet the experimental section only compares the proposed model against LSTM, PatchTST, and Simple-Mamba. Why were these reconstruction-based models excluded from direct empirical benchmarking?

**Draft Author Response:**
> We agree with the reviewer that these models are key references in anomaly detection. However, direct empirical comparison against reconstruction-based models (e.g., USAD, TranAD, OmniAnomaly) was intentionally avoided due to fundamental differences in optimization objectives and validation procedures:
>
> 1. **Paradigm Incompatibility:** Reconstruction models optimize autoencoding loops to reconstruct the current sequence, whereas the proposed model operates under the Long-term Forecasting (LTF) paradigm. These represent different mathematical spaces and loss functions, making direct numerical comparison of prediction errors invalid.
> 2. **Over-generalization Risk:** Deep reconstruction networks are highly vulnerable to reconstructing early-stage fault pulses (as the network over-generalizes), which leads to missed alarms. In contrast, forecasting models rely on temporal causality, which is immediately broken by fault onset.
> 3. **Validation Data Leakage:** Most reconstruction baselines (such as TranAD and OmniAnomaly) tune their decision thresholds globally using validation datasets that contain future anomalies. This induces severe data leakage. Our paper restricts its benchmarking strictly to forecasting models under a **leakage-free** constraint (threshold calibrated solely on early healthy data).

**Giải thích chiến lược (Vietnamese):**
* Nhấn mạnh sự khác biệt về bản chất giữa mô hình **Tái cấu trúc (Reconstruction)** và mô hình **Dự báo (Forecasting)**.
* Chỉ ra hai điểm yếu cốt lõi của dòng tái cấu trúc: Dễ bị quá tổng quát hóa gây bỏ sót lỗi giai đoạn đầu, và đặc biệt là bị **rò rỉ dữ liệu (data leakage)** khi xác định ngưỡng bằng dữ liệu lỗi tương lai (vận hành thực tế online không bao giờ làm được như vậy).

---

## 5. Setting All Models to ~338k Parameters (Ràng buộc tham số ~338k)

**Reviewer Comment:**
> All baseline models were restricted to a parameter size of approximately 338k. This fixed parameter constraint might handicap models like PatchTST or LSTM, which may require larger architectures to perform optimally. Please justify this choice.

**Draft Author Response:**
> We thank the reviewer for this important point. The parameter size constraint was implemented to simulate a strict edge-device footprint and ensure a fair, rigorous evaluation of architectural efficiency:
>
> 1. **Edge-Computing Limitations:** The ~338k parameter budget corresponds to the memory bounds of standard edge-computing hardware (e.g., ARM Cortex-M7 microcontrollers and Edge TPUs) commonly deployed for real-time machinery diagnostics.
> 2. **Hyperparameter Optimization Parity:** To ensure the baselines were not unfairly handicapped, we performed an automated grid search/hyperparameter sweep for each baseline model individually under the 338k parameter constraint. We optimized layer depths, hidden units, patch sizes, and learning rates to ensure that each baseline model achieved its peak, optimal performance *within* the parameter budget.
> 3. **Scalability Trade-off:** Even if PatchTST were scaled up to millions of parameters, its memory footprint (already at 1335.4 MB for BS=64) would immediately exceed edge memory limits, making the comparison irrelevant for real-world edge diagnostics.

**Giải thích chiến lược (Vietnamese):**
* Giải thích rằng ~338k là ngưỡng phần cứng thiết bị biên (ARM Cortex-M/Edge TPU).
* Khẳng định chúng ta đã chạy tìm kiếm siêu tham số (grid search) tối ưu cho từng baseline dưới ràng buộc này, nên chúng đã chạy ở mức đỉnh của chúng chứ không bị dìm hiệu năng.

---

## 6. Training Epochs & Convergence (Thời gian huấn luyện 10 Epochs)

**Reviewer Comment:**
> The models were trained for only 10 epochs, which seems arbitrary and potentially insufficient for complete model convergence. Please provide convergence curves and justify this training duration.

**Draft Author Response:**
> We appreciate this comment. The 10-epoch training duration is mathematically justified by the loss convergence curves presented in **Figure 2**.
>
> 1. **Empirical Convergence:** As shown in Figure 2, both the training and validation Huber losses for all architectures reach a stable, flat asymptotic plateau by epoch 6–8. 
> 2. **Rapid Learning Dynamics:** Because the training phase is conducted exclusively on stationary, clean healthy operational segments (the first 45% of the lifecycle), and we employ an optimized Adam optimizer ($5 \times 10^{-4}$), the models capture the nominal dynamics very rapidly.
> 3. **Risk of Overfitting:** Continuing training past 10 epochs does not reduce the validation loss further, but increases the risk of overfitting the healthy baseline fluctuations, which reduces the downstream sensitivity of the model to early structural degradation faults.

**Giải thích chiến lược (Vietnamese):**
* Dẫn chứng trực tiếp vào **Hình 2** (loss convergence curve) chứng minh loss đã phẳng từ epoch 6-8.
* Giải thích do dữ liệu huấn luyện là pha khỏe mạnh ổn định nên học rất nhanh. Chạy thêm epoch chỉ gây quá khớp (overfit) pha khỏe mạnh và làm giảm độ nhạy phát hiện lỗi về sau.
