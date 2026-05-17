import os
import torch
import pandas as pd
from tqdm import tqdm
import argparse

def preprocess_pronostia(raw_dir, processed_base_dir):
    """
    Chuyển đổi dataset PRONOSTIA (IEEE PHM 2012) từ định dạng .csv sang .pt (Tensor).
    - Cột 4: Horizontal Acceleration
    - Cột 5: Vertical Acceleration
    """
    subsets = ['Learning_set', 'Full_Test_Set']
    
    for subset in subsets:
        subset_dir = os.path.join(raw_dir, subset)
        if not os.path.exists(subset_dir):
            continue
            
        bearings = sorted([d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))])
        
        for bearing in bearings:
            bearing_dir = os.path.join(subset_dir, bearing)
            output_dir = os.path.join(processed_base_dir, f"PRONOSTIA_{bearing}")
            os.makedirs(output_dir, exist_ok=True)
            
            csv_files = sorted([f for f in os.listdir(bearing_dir) if f.startswith('acc_') and f.endswith('.csv')])
            print(f"\n[{subset}] Đang xử lý {bearing}: {len(csv_files)} files...")
            
            for f in tqdm(csv_files):
                file_path = os.path.join(bearing_dir, f)
                try:
                    # Đọc cột 4 (Horiz) và cột 5 (Vert). Lưu ý: pandas index từ 0
                    try:
                        df = pd.read_csv(file_path, header=None, usecols=[4, 5])
                    except ValueError:
                        # Full_Test_Set thường dùng dấu chấm phẩy thay vì dấu phẩy
                        df = pd.read_csv(file_path, header=None, sep=';', usecols=[4, 5])
                    
                    # Numpy array shape: (2560, 2)
                    arr = df.values.astype('float32')
                    
                    # Transpose to (2, 2560) để phù hợp kiến trúc mô hình
                    tensor_data = torch.from_numpy(arr.T)
                    
                    save_name = f.replace('.csv', '.pt')
                    torch.save(tensor_data, os.path.join(output_dir, save_name))
                except Exception as e:
                    print(f"Lỗi khi đọc {file_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='data/raw/ieee-phm-2012-data-challenge-dataset-master')
    parser.add_argument('--processed_dir', type=str, default='data/processed')
    args = parser.parse_args()
    
    print(f"Bắt đầu tiền xử lý PRONOSTIA từ {args.raw_dir}")
    preprocess_pronostia(args.raw_dir, args.processed_dir)
    print("\nHoàn tất! Các file .pt đã được lưu vào", args.processed_dir)
