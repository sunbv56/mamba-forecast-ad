import argparse
import yaml
import sys
import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Hybrid Mamba-CNN for B02 Dataset")
    parser.add_argument('--config', type=str, default='configs/pretrain_b02.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Bắt đầu với chế độ: {args.mode}")
    print(f"Cấu hình: {config}")

    if not torch.cuda.is_available():
        print("Cảnh báo: Không tìm thấy GPU, hệ thống Mamba yêu cầu CUDA để tối ưu.")

    if args.mode == 'train':
        print("Đang khởi tạo Trainer...")
        # trainer = Trainer(config)
        # trainer.train()
    elif args.mode == 'eval':
        print("Đang chạy Evaluation...")

if __name__ == "__main__":
    main()
