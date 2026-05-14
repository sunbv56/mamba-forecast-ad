import os
import yaml
import argparse
import subprocess
import time
import sys

def run_command(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    start_time = time.time()
    result = subprocess.run(cmd)
    duration = time.time() - start_time
    print(f">>> Finished in {duration:.2f}s (Exit code: {result.returncode})")
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Master Runner for Multi-bearing Anomaly Detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing for all datasets")
    parser.add_argument("--train", action="store_true", help="Run training on joint datasets")
    parser.add_argument("--eval", action="store_true", help="Run full life-cycle evaluation for all bearings")
    parser.add_argument("--model", type=str, default="Mamba1-Hybrid", help="Model name to train/eval")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    train_datasets = config['data'].get('train_datasets', [config['data'].get('processed_dir')])
    test_datasets = config['data'].get('test_datasets', [config['data'].get('processed_dir')])
    all_datasets = list(set(train_datasets + test_datasets))
    
    # 1. Preprocessing
    if args.preprocess:
        print("\n" + "="*60)
        print("PHASE 1: PREPROCESSING")
        print("="*60)
        for ds_path in all_datasets:
            # Giả định raw_dir tương ứng (ví dụ data/processed/B02 -> data/raw/B02)
            ds_name = os.path.basename(ds_path)
            raw_dir = os.path.join("data/raw", ds_name)
            if os.path.exists(raw_dir):
                cmd = [sys.executable, "src/data/pipeline.py", "--config", args.config, "--raw_dir", raw_dir, "--processed_dir", ds_path]
                run_command(cmd)
            else:
                print(f"Warning: Raw directory {raw_dir} not found for {ds_path}")

    # 2. Training
    if args.train:
        print("\n" + "="*60)
        print("PHASE 2: TRAINING")
        print("="*60)
        cmd = [sys.executable, "src/training/train.py", "--config", args.config, "--model", args.model]
        run_command(cmd)

    # 3. Evaluation & Visualization
    if args.eval:
        print("\n" + "="*60)
        print("PHASE 3: EVALUATION & VISUALIZATION")
        print("="*60)
        
        # Tìm model file mới nhất
        model_name_safe = args.model.lower().replace("-", "_")
        model_path = f"results/models/{model_name_safe}_best.pth"
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return

        for ds_path in test_datasets:
            print(f"\n--- Visualizing {ds_path} ---")
            # Cần tạm thời cập nhật config hoặc truyền processed_dir qua CLI
            # Ở đây visualize_full_lifecycle.py đọc từ config['data']['processed_dir']
            # Ta sẽ tạo một file config tạm hoặc dùng sed để thay thế (đơn giản nhất là truyền vào CLI nếu script hỗ trợ)
            
            # Cập nhật visualize_full_lifecycle.py để nhận --processed_dir CLI? 
            # Hoặc tạo config tạm:
            temp_config_path = f"configs/temp_eval_{os.path.basename(ds_path)}.yaml"
            temp_config = config.copy()
            temp_config['data']['processed_dir'] = ds_path
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_config, f)
            
            cmd = [sys.executable, "src/evaluation/visualize_full_lifecycle.py", "--model_path", model_path, "--config", temp_config_path]
            run_command(cmd)
            
            # Xóa config tạm
            os.remove(temp_config_path)

    print("\n" + "="*60)
    print("ALL PHASES COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
