import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal


class BearingDataset(Dataset):
    def __init__(
        self,
        data_dir,
        lookback=1024,
        horizon=64,
        stride=512,
        split='train',
        normalize=True,
        file_sample_ratio=1,
        fault_rms_factor=3.0,
        train_rms_pct=40,
        test_rms_pct=80,
        oc_stats=None,
        train_ratio=0.5,
        skip_ratio=0.1,
        highpass_freq=0,
        sampling_rate=128000
    ):
        """
        DataLoader cho bộ dữ liệu vòng bi. Trả về cặp (X, Y, stats, label) để Forecasting + AD.
        """
        self.data_dir        = data_dir
        self.lookback        = lookback
        self.horizon         = horizon
        self.stride          = stride
        self.split           = split
        self.normalize       = normalize
        self.file_sample_ratio = file_sample_ratio
        self.fault_rms_factor  = fault_rms_factor
        self.train_rms_pct   = train_rms_pct
        self.test_rms_pct    = test_rms_pct
        self.oc_stats        = oc_stats
        self.train_ratio     = train_ratio
        self.skip_ratio      = skip_ratio
        self.highpass_freq   = highpass_freq
        self.sampling_rate   = sampling_rate

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        
        oc_path = os.path.join(data_dir, 'operating_conditions.csv')
        if os.path.exists(oc_path):
            self.oc_df = pd.read_csv(oc_path)
            
            # [NEW] Chuẩn hóa OC (Operating Conditions) thống nhất
            oc_cols = self.oc_df.columns[1:] # Bỏ cột Time
            oc_values = self.oc_df[oc_cols].values
            
            if self.oc_stats is None:
                # Nếu chưa có stats (thường là ở tập Train), tính toán mới
                self.oc_stats = {
                    'mean': oc_values.mean(axis=0),
                    'std': oc_values.std(axis=0) + 1e-6
                }
            
            self.oc_df[oc_cols] = (oc_values - self.oc_stats['mean']) / self.oc_stats['std']
        else:
            self.oc_df = None

        # Load hoặc compute per-file RMS (cached)
        self.file_rms = self._load_or_compute_file_rms()

        # Healthy baseline = P10 của các file trong vùng Healthy dự kiến (vùng Train)
        n_files = len(self.files)
        skip_end = int(n_files * self.skip_ratio)
        train_end = int(n_files * (self.skip_ratio + self.train_ratio))
        
        healthy_files_for_baseline = self.files[skip_end:train_end]
        all_rms = np.array([self.file_rms[f] for f in healthy_files_for_baseline])
        self.healthy_rms_baseline = float(np.percentile(all_rms, 10)) if len(all_rms) > 0 else 0.01

        self.samples = []
        self.signal_cache = {}
        self._prepare_samples()

    def _load_or_compute_file_rms(self):
        """Load file_rms.json nếu đã có, ngược lại tính và lưu cache."""
        rms_cache_path = os.path.join(self.data_dir, 'file_rms.json')
        if os.path.exists(rms_cache_path):
            with open(rms_cache_path, 'r') as f:
                return json.load(f)

        print(f"Computing per-file RMS for {os.path.basename(self.data_dir)}...")
        file_rms = {}
        for i, fname in enumerate(self.files):
            fpath = os.path.join(self.data_dir, fname)
            data = torch.load(fpath, weights_only=True)
            rms = float(data.pow(2).mean().sqrt().item())
            file_rms[fname] = rms

        with open(rms_cache_path, 'w') as f:
            json.dump(file_rms, f, indent=2)
        return file_rms

    def _compute_physical_stats(self, x):
        """Tính toán 8 đặc trưng thống kê vật lý trên tín hiệu thô x (Shape: [C, L])"""
        eps = 1e-8
        mean       = x.mean(dim=-1, keepdim=True)
        std        = x.std(dim=-1, keepdim=True) + eps
        rms        = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        peak2peak  = x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0]
        skewness   = torch.mean(((x - mean) / std) ** 3, dim=-1, keepdim=True)
        kurtosis   = torch.mean(((x - mean) / std) ** 4, dim=-1, keepdim=True)
        crest_factor = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] / (rms + eps)
        shape_factor = rms / (torch.mean(torch.abs(x), dim=-1, keepdim=True) + eps)
        stats = torch.cat(
            [mean, std, rms, peak2peak, skewness, kurtosis, crest_factor, shape_factor],
            dim=-1
        )  # (C, 8)
        return stats

    def _prepare_samples(self):
        if not self.files:
            return

        # Đọc 1 file để lấy độ dài
        sample_data = torch.load(os.path.join(self.data_dir, self.files[0]), weights_only=True)
        n_samples_per_file = sample_data.shape[1]

        window_size      = self.lookback + self.horizon
        n_windows_per_file = (n_samples_per_file - window_size) // self.stride + 1

        n_files = len(self.files)
        skip_end  = int(n_files * self.skip_ratio)
        train_end = int(n_files * (self.skip_ratio + self.train_ratio))
        
        healthy_indices = list(range(skip_end, train_end))
        faulty_indices  = list(range(train_end, n_files))
        
        if self.split == 'train':
            file_indices = healthy_indices[0::2]
        elif self.split == 'val':
            file_indices = healthy_indices[1::4]
        else: # test
            train_val_indices = set(healthy_indices[0::2]) | set(healthy_indices[1::4])
            remaining_healthy = [i for i in healthy_indices if i not in train_val_indices]
            file_indices = remaining_healthy + faulty_indices

        if self.file_sample_ratio > 1 and self.split != 'test':
            file_indices = file_indices[::self.file_sample_ratio]

        for f_idx in file_indices:
            for w_idx in range(n_windows_per_file):
                self.samples.append((f_idx, w_idx * self.stride))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_idx, offset = self.samples[idx]

        if f_idx not in self.signal_cache:
            file_path = os.path.join(self.data_dir, self.files[f_idx])
            sig = torch.load(file_path, weights_only=True)
            
            if self.highpass_freq > 0:
                nyq = 0.5 * self.sampling_rate
                normal_cutoff = self.highpass_freq / nyq
                b, a = scipy_signal.butter(4, normal_cutoff, btype='high', analog=False)
                sig_np = sig.numpy()
                sig_filtered = scipy_signal.lfilter(b, a, sig_np, axis=1)
                sig = torch.from_numpy(sig_filtered.copy()).float()
            
            self.signal_cache[f_idx] = sig

        signal = self.signal_cache[f_idx]

        end_x = offset + self.lookback
        end_y = end_x + self.horizon

        x_raw = signal[:, offset:end_x]
        y     = signal[:, end_x:end_y]

        stats = self._compute_physical_stats(x_raw)
        x = x_raw.clone()

        current_file_rms = self.file_rms[self.files[f_idx]]
        if current_file_rms > self.healthy_rms_baseline * self.fault_rms_factor:
            label = 1
        else:
            label = 0

        if self.normalize:
            mean_x = x.mean(dim=1, keepdim=True)
            std_x  = x.std(dim=1, keepdim=True) + 1e-8
            x = (x - mean_x) / std_x
            y = (y - mean_x) / std_x

        if self.oc_df is not None:
            oc = self.oc_df.iloc[f_idx].values[1:].astype('float32')
            oc = torch.from_numpy(oc)
            return x, y, stats, label, oc

        return x, y, stats, label

# Alias for backward compatibility
B02Dataset = BearingDataset

class MultiBearingDataset(Dataset):
    def __init__(self, data_dirs, **kwargs):
        """
        Gộp nhiều bộ dữ liệu vòng bi từ nhiều thư mục.
        """
        self.datasets = []
        oc_stats = kwargs.get('oc_stats', None)
        
        for d in data_dirs:
            # Truyền oc_stats từ dataset đầu tiên sang các dataset sau (nếu có)
            ds = BearingDataset(data_dir=d, **kwargs)
            if oc_stats is None and ds.oc_df is not None:
                oc_stats = ds.oc_stats
                kwargs['oc_stats'] = oc_stats
            self.datasets.append(ds)
            
        self.length = sum(len(ds) for ds in self.datasets)
        
        # Cumulative indices for mapping
        self.cumulative_lengths = [0]
        curr = 0
        for ds in self.datasets:
            curr += len(ds)
            self.cumulative_lengths.append(curr)
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i in range(len(self.cumulative_lengths) - 1):
            if idx < self.cumulative_lengths[i+1]:
                return self.datasets[i][idx - self.cumulative_lengths[i]]
        raise IndexError("Index out of bounds")

    @property
    def oc_stats(self):
        # Lấy oc_stats từ dataset đầu tiên có OC
        for ds in self.datasets:
            if ds.oc_stats is not None:
                return ds.oc_stats
        return None

