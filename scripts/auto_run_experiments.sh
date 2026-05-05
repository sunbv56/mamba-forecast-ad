#!/bin/bash

# Script tự động chạy Grid Search qua nhiều cấu hình mô hình, lookback, horizon
# Dành cho Sinh viên AI 2

LOOKBACKS=(512 1024 2048)
HORIZONS=(256 512)
MODELS=("lstm" "tcn" "transformer_small")

for model in "${MODELS[@]}"; do
    for N in "${LOOKBACKS[@]}"; do
        for K in "${HORIZONS[@]}"; do
            echo "=========================================="
            echo "Running Experiment: Model=$model, Lookback=$N, Horizon=$K"
            echo "=========================================="
            
            # Dưới đây là lệnh mẫu gọi main.py để chạy pipeline
            # python main.py --model_type $model --lookback $N --horizon $K --epochs 10 --batch_size 32
            
            # (Bạn có thể bỏ comment dòng lệnh trên khi đã viết xong file main.py nhận tham số)
        done
    done
done
echo "Hoàn thành Grid Search."
