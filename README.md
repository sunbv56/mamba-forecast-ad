# Mamba-Forecast-AD: Anomaly Detection via Time-Series Forecasting

This project implements an **Anomaly Detection (AD)** system for industrial bearing data using **Time-Series Forecasting**. It leverages the **Mamba** (Selective State Space Model) architecture alongside traditional baselines to predict future signals and identify anomalies based on forecasting errors.

## 🔗 Project Links

- **Kaggle Notebook (Implementation & Experiments):** [Mamba-Forecast-AD on Kaggle](https://www.kaggle.com/code/sunbv21/mamba-forecast-ad)
- **Dataset (Zenodo):** [Bearing Failure Dataset (B02)](https://zenodo.org/doi/10.5281/zenodo.10805042)

## 📌 Project Overview

The core idea is to train a model on "healthy" data to learn normal operating patterns. During inference, the model forecasts the next window of signals. If the forecasting error (Anomaly Score) exceeds a statistical threshold (e.g., 3-sigma), an anomaly is detected, signaling potential bearing degradation or failure.

### Key Features
- **Hybrid Mamba-CNN Architecture:** Combines the long-range dependency modeling of Mamba with the local feature extraction of CNNs.
- **Baseline Models:** Includes LSTM, TCN, and Patch-based Transformer for performance benchmarking.
- **Automated Pipeline:** Full pipeline from raw data windowing/normalization to automated experiment execution.
- **Rich Visualization:** Generates Anomaly Score charts, TTF (Time To Failure) analysis, and Confusion Matrices.

## 🌟 Scientific Value & Novelty

Compared to existing literature on Mamba for bearing analysis, this project stands out by focusing on:

1.  **Unsupervised Learning (Self-Supervised Forecasting):** Unlike most papers that require failure labels for direct RUL regression or fault classification, our model learns healthy patterns from normal operating data. This is significantly more practical for industrial applications where failure data is scarce.
2.  **Anomaly Score as a Degradation Indicator:** We utilize the forecasting error (MSE) as a dynamic Anomaly Score. This score provides a continuous health index that naturally increases as the bearing deviates from its learned "healthy" state, offering a physically intuitive way to monitor degradation without explicit stage labels.
3.  **Modern Dataset (B02 Zenodo):** Experiments are conducted on the recent B02 dataset (2024), providing fresh insights beyond the over-used PRONOSTIA and CWRU benchmarks.

## 📁 Project Structure

```text
├── data/               # Raw and processed datasets
├── configs/            # YAML configuration files for experiments
├── src/                # Main source code
│   ├── data/           # Dataset loading and preprocessing pipeline
│   ├── models/         # Mamba and baseline implementations (LSTM, TCN, Transformer)
│   ├── training/       # Training loops and Trainer class
│   ├── evaluation/     # Anomaly scoring and metric calculation
│   └── utils/          # Visualization and logging utilities
├── scripts/            # Shell scripts for automated runs
├── results/            # Saved models, logs, and plots
├── main.py             # Entry point for training and evaluation
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-repo/mamba-forecast-ad.git
cd mamba-forecast-ad
pip install mamba-ssm --no-build-isolation
pip install -r requirements.txt
```

*Note: Mamba requires a CUDA-enabled environment for optimal performance.*

### 2. Dataset Preparation

Download the dataset B02.zip from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10805042) and place it in the `data/raw/` directory.

### 3. Usage

The project is designed to be run through Jupyter Notebooks for interactive exploration and training:

1.  **Data Exploration:** Open `notebooks/01_B02_Data_Exploration.ipynb` to understand the dataset structure, visualize signals, and analyze failure timestamps.
2.  **Training & Evaluation:** Use `notebooks/mamba-forecast-ad.ipynb` to run the full pipeline, including training the Hybrid Mamba-CNN model and evaluating its anomaly detection performance.

## 📊 Evaluation Metrics
- **Anomaly Score:** Mean Squared Error (MSE) between predicted and actual signals.
- **Detection Delay:** Time difference between actual failure and model detection.
- **False Alarm Rate (FAR)**
- **F1-Score / AUC**