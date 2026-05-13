# MambaTS: Pipeline Logic & Mathematical Flow

This document details the mathematical and logical operations within the three core pipelines of the MambaTS anomaly detection system.

---

## 1. Data Processing Logic

### Windowing & Forecasting Pairs
For a given signal $S \in \mathbb{R}^{C \times L_{total}}$, where $C$ is the number of channels, the dataset generates samples $(X, Y)$:
- **Input (Lookback)**: $X = S[:, t:t+L] \in \mathbb{R}^{C \times L}$
- **Target (Horizon)**: $Y = S[:, t+L:t+L+K] \in \mathbb{R}^{C \times K}$
- $L$: Lookback length, $K$: Horizon length.

### Normalization (RevIN)
To handle non-stationarity, each window $X$ is normalized:
$$\mu_X = \frac{1}{L} \sum_{i=1}^L X_i, \quad \sigma_X = \sqrt{\frac{1}{L} \sum_{i=1}^L (X_i - \mu_X)^2 + \epsilon}$$
$$\hat{X} = \frac{X - \mu_X}{\sigma_X}$$
The same $\mu_X$ and $\sigma_X$ are used to normalize the target $Y$ and denormalize the prediction $\hat{Y}$.

---

## 2. Model Architectural Logic

### 2.1. MambaTS (Pure/Official) Logic

#### Patching
The normalized input $\hat{X}$ is divided into $N$ patches of size $P$ with stride $S$:
$$N = \lfloor \frac{L - P}{S} \rfloor + 1$$
Each patch is projected: $E_{p} = \text{Linear}(\text{Patch}) \in \mathbb{R}^D$.

#### Variable-Aware Scanning (VAS)
1. **Correlation Matrix**: Calculate $R \in \mathbb{R}^{C \times C}$ where $R_{ij}$ is the Pearson correlation between channel $i$ and $j$.
2. **ATSP Solver**: Find a permutation $\pi$ of $\{1, \dots, C\}$ that maximizes $\sum R_{\pi(k), \pi(k+1)}$.
3. **VST**: Arrange tokens in the order $(\text{Var}_{\pi(1)}, \text{Patch}_1, \dots, \text{Patch}_N, \text{Var}_{\pi(2)}, \dots)$.

#### Selective SSM (Mamba)
The sequence of tokens $T$ is processed via:
$$h_t = \mathbf{A}h_{t-1} + \mathbf{B}_t x_t$$
$$y_t = \mathbf{C}_t h_t + \mathbf{D}x_t$$
Where $\mathbf{B}_t, \mathbf{C}_t,$ and $\Delta_t$ are functions of the input $x_t$.

### 2.2. HybridMamba (CI-Mamba++) Logic

#### Series Decomposition
Input $X$ is decomposed using a moving average kernel of size $k$:
$$X_{trend} = \text{AvgPool}(X, \text{kernel}=k)$$
$$X_{seasonal} = X - X_{trend}$$

#### Multi-Scale Patching
The seasonal component is embedded using multiple patch sizes $\{P_1, P_2, \dots, P_m\}$ to capture diverse frequency components:
$$E_{seasonal} = \text{Concat}(\text{PatchEmbed}_{P_1}(X_{seasonal}), \dots, \text{PatchEmbed}_{P_m}(X_{seasonal}))$$

#### Forecast Mixing
The final forecast $\hat{Y}$ is a weighted sum of the seasonal forecast $\hat{Y}_s$ and trend forecast $\hat{Y}_t$:
$$\hat{Y} = \sigma(\alpha) \cdot \hat{Y}_s + (1 - \sigma(\alpha)) \cdot \hat{Y}_t$$
Where $\alpha$ is a learnable parameter per channel.

> [!TIP]
> **Performance Note**: HybridMamba (CI-Mamba++) demonstrates superior performance in bearing fault diagnosis compared to pure MambaTS. By explicitly modeling the global trend via the Linear branch and local oscillations via Multi-scale Mamba, it achieves significantly lower reconstruction error and faster convergence on high-frequency vibration signals.

---

## 3. Anomaly Detection & Evaluation Logic

### Anomaly Score
The anomaly score $s$ for a window is the Mean Squared Error of the prediction:
$$s = \frac{1}{C \cdot K} \sum_{c=1}^C \sum_{k=1}^K (Y_{c,k} - \hat{Y}_{c,k})^2$$
Normalized Score: $s_{norm} = \ln(1 + s)$ (optional).

### Thresholding (POT - Peak Over Threshold)
Based on Extreme Value Theory (EVT), we model the "tail" of the error distribution:
1. Choose an initial high quantile $t$ (e.g., P98).
2. Fit a Generalized Pareto Distribution (GPD) to the excesses $E = \{s_i - t \mid s_i > t\}$.
3. Calculate the threshold $z_q$ for a target probability $q$ (e.g., $10^{-4}$):
$$z_q = t + \frac{\sigma}{\gamma} \left( \left( \frac{n}{N_t} q \right)^{-\gamma} - 1 \right)$$

### Evaluation Metrics
- **F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **FAR (False Alarm Rate)**: $\frac{FP}{FP + TN}$
- **Detection Delay**: $T_{detection} - T_{fault\_onset}$
  - $T_{fault\_onset}$ is defined as the first time index where the physical RMS exceeds $3 \times$ the healthy baseline.
