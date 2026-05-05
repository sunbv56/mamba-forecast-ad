#!/bin/bash

# Kích hoạt môi trường ảo nếu có
# source ../venv/bin/activate

echo "Bắt đầu tiền xử lý dữ liệu B02..."
python -m src.data.preprocess --config configs/pretrain_b02.yaml
