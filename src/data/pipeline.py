import argparse
import os
import scipy.io as sio
import torch
import pandas as pd
from tqdm import tqdm
import yaml

def preprocess_b02(config):
    """
    Đọc tất cả các file .mat trong raw_dir, 
    áp dụng hệ số chuyển đổi (10 g/V) và lưu lại dưới dạng .pt tại processed_dir.
    """
    raw_dir = config['data'].get('raw_dir', 'data/raw/B02')
    processed_dir = config['data'].get('processed_dir', 'data/processed')
    vibration_dir = os.path.join(raw_dir, 'vibrationData')
    
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Đang đọc dữ liệu thô từ: {raw_dir}")
    
    # Load and save operating conditions
    oc_path = os.path.join(raw_dir, 'B02_operatingConditions.csv')
    if os.path.exists(oc_path):
        oc_df = pd.read_csv(oc_path)
        oc_df.to_csv(os.path.join(processed_dir, 'operating_conditions.csv'), index=False)
        print(f"Đã lưu operating conditions vào {processed_dir}")
    else:
        print(f"Cảnh báo: Không tìm thấy {oc_path}")

    # Chuyển đổi file .mat sang .pt
    mat_files = sorted([f for f in os.listdir(vibration_dir) if f.endswith('.mat')])
    print(f"Tìm thấy {len(mat_files)} file .mat. Bắt đầu chuyển đổi...")
    
    conversion_factor = 10.0 # 10 g/V
    
    for filename in tqdm(mat_files):
        file_path = os.path.join(vibration_dir, filename)
        mat_data = sio.loadmat(file_path)
        
        signal_frontal = mat_data['accHorizFrontal_C'].astype('float32') * conversion_factor
        signal_rear = mat_data['accHorizRear_A'].astype('float32') * conversion_factor
        
        signal_frontal_t = torch.from_numpy(signal_frontal.T) # (1, N)
        signal_rear_t = torch.from_numpy(signal_rear.T) # (1, N)
        signal = torch.cat([signal_frontal_t, signal_rear_t], dim=0) # (2, N)
        
        save_name = filename.replace('.mat', '.pt')
        torch.save(signal, os.path.join(processed_dir, save_name))

    print(f"Hoàn thành! Dữ liệu Tensor đã được lưu tại: {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocess_b02(config)
