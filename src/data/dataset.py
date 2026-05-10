import torch
from torch.utils.data import Dataset
import os
import pandas as pd

class B02Dataset(Dataset):
    def __init__(self, data_dir, lookback=1024, horizon=512, stride=512, split='train', normalize=True):
        """
        DataLoader cho bộ dữ liệu B02. Trả về cặp (X, Y) để Forecasting.
        Args:
            data_dir: Đường dẫn đến thư mục chứa các file .pt
            lookback (N): Độ dài chuỗi quá khứ
            horizon (K): Độ dài chuỗi tương lai cần dự báo
            stride: Bước nhảy của cửa sổ trượt
            split: 'train', 'val', hoặc 'test'
            normalize: Có chuẩn hóa dữ liệu hay không
        """
        self.data_dir = data_dir
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.split = split
        self.normalize = normalize
        
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        
        oc_path = os.path.join(data_dir, 'operating_conditions.csv')
        if os.path.exists(oc_path):
            self.oc_df = pd.read_csv(oc_path)
        else:
            self.oc_df = None

        self.samples = []
        self.signal_cache = {} # Lưu trữ signal đã load để tránh đọc ổ cứng liên tục
        self._prepare_samples()

    def _compute_physical_stats(self, x):
        """Tính toán 8 đặc trưng thống kê vật lý trên tín hiệu thô x (Shape: [C, L])"""
        eps = 1e-8
        
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + eps
        
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        peak2peak = x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0]
        
        skewness = torch.mean(((x - mean) / std)**3, dim=-1, keepdim=True)
        kurtosis = torch.mean(((x - mean) / std)**4, dim=-1, keepdim=True)
        
        crest_factor = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] / (rms + eps)
        shape_factor = rms / (torch.mean(torch.abs(x), dim=-1, keepdim=True) + eps)
        
        # Nối tất cả lại thành tensor shape [C, 8]
        stats = torch.cat([mean, std, rms, peak2peak, skewness, kurtosis, crest_factor, shape_factor], dim=-1)
        return stats

    def _prepare_samples(self):
        # ... (giữ nguyên logic cũ)
        if not self.files:
            return
            
        # Đọc 1 file để lấy độ dài
        sample_file = torch.load(os.path.join(self.data_dir, self.files[0]), weights_only=True)
        n_samples_per_file = sample_file.shape[1]
        
        window_size = self.lookback + self.horizon
        n_windows_per_file = (n_samples_per_file - window_size) // self.stride + 1
        
        n_files = len(self.files)
        # Phân chia theo file: 80% train, 10% val, 10% test
        if self.split == 'train':
            file_indices = range(0, int(0.8 * n_files))
        elif self.split == 'val':
            file_indices = range(int(0.8 * n_files), int(0.9 * n_files))
        else:
            file_indices = range(int(0.9 * n_files), n_files)

        for f_idx in file_indices:
            for w_idx in range(n_windows_per_file):
                self.samples.append((f_idx, w_idx * self.stride))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_idx, offset = self.samples[idx]
        
        if f_idx not in self.signal_cache:
            file_path = os.path.join(self.data_dir, self.files[f_idx])
            self.signal_cache[f_idx] = torch.load(file_path, weights_only=True)
            
        signal = self.signal_cache[f_idx] # (2, N)
        
        end_x = offset + self.lookback
        end_y = end_x + self.horizon
        
        # X: quá khứ, Y: tương lai
        x_raw = signal[:, offset : end_x]
        y = signal[:, end_x : end_y]
        
        # Tính toán đặc trưng vật lý trước khi chuẩn hóa
        stats = self._compute_physical_stats(x_raw)
        x = x_raw.clone() # Tránh in-place modification làm ảnh hưởng stats
        
        if self.normalize:
            # Dùng Z-score dựa trên mean/std của X để chuẩn hóa cả X và Y (tránh rò rỉ dữ liệu)
            mean_x = x.mean(dim=1, keepdim=True)
            std_x = x.std(dim=1, keepdim=True) + 1e-8
            x = (x - mean_x) / std_x
            y = (y - mean_x) / std_x
            
        if self.oc_df is not None:
            oc = self.oc_df.iloc[f_idx].values[1:].astype('float32') # Bỏ cột thời gian
            oc = torch.from_numpy(oc)
            return x, y, stats, oc
            
        return x, y, stats
