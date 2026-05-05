#!/bin/bash

# Kích hoạt môi trường ảo nếu có
# source ../venv/bin/activate

echo "Bắt đầu tiến trình huấn luyện Hybrid Mamba-CNN với dữ liệu B02..."
python main.py --config configs/pretrain_b02.yaml --mode train
