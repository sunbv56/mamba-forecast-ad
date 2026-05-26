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
        normalize=False,
        file_sample_ratio=1,
        fault_rms_factor=3.0,
        train_rms_pct=40,
        test_rms_pct=80,
        oc_stats=None,
        train_ratio=0.5,
        skip_ratio=0.1,
        highpass_freq=0,
        sampling_rate=128000,
        label_strategy='rms',
        manual_fault_start=None,
        **kwargs
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
        self.label_strategy  = label_strategy
        self.manual_fault_start = manual_fault_start

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        
        oc_path = os.path.join(data_dir, 'operating_conditions.csv')
        is_pronostia = 'pronostia' in data_dir.lower() or 'phm' in data_dir.lower()
        
        if os.path.exists(oc_path):
            self.oc_df = pd.read_csv(oc_path)
            
            # [NEW] Chuẩn hóa OC: Chỉ trích xuất cột Speed và Load
            speed_col = next((c for c in self.oc_df.columns if 'setspeed' in c.lower()), None)
            load_col = next((c for c in self.oc_df.columns if 'setdynload' in c.lower()), None)
            
            if speed_col and load_col:
                oc_cols = [speed_col, load_col]
            else:
                oc_cols = self.oc_df.columns[1:3] # Fallback
                
            oc_values = self.oc_df[oc_cols].values
            
            if self.oc_stats is None:
                # Nếu chưa có stats (thường là ở tập Train), tính toán mới
                self.oc_stats = {
                    'mean': oc_values.mean(axis=0),
                    'std': oc_values.std(axis=0) + 1e-6
                }
            
            self.oc_values_processed = (oc_values - self.oc_stats['mean']) / self.oc_stats['std']
        elif is_pronostia:
            self.oc_df = None
            oc_values = np.array([[1800.0, 4000.0]]) # Hardcode cho PRONOSTIA (1 sample)
            
            if self.oc_stats is None:
                self.oc_stats = {
                    'mean': oc_values.mean(axis=0),
                    'std': oc_values.std(axis=0) + 1e-6
                }
                
            static_oc = (np.array([1800.0, 4000.0]) - self.oc_stats['mean']) / self.oc_stats['std']
            self.static_oc = static_oc.astype('float32')
            self.oc_values_processed = None
        else:
            self.oc_df = None
            self.oc_values_processed = None
            self.static_oc = None

        # Load hoặc compute per-file RMS (cached)
        self.file_rms = self._load_or_compute_file_rms()

        # [NEW] Làm mịn RMS bằng Rolling Mean (window=10) để gán nhãn ổn định hơn, tránh nhiễu ảo cục bộ
        raw_rms_array = np.array([self.file_rms[f] for f in self.files])
        smoothed_rms_array = pd.Series(raw_rms_array).rolling(window=10, min_periods=1, center=True).mean().values
        self.smoothed_file_rms = {f: smoothed_rms_array[i] for i, f in enumerate(self.files)}

        # Healthy baseline = P10 của các file trong vùng Healthy dự kiến (vùng Train)
        n_files = len(self.files)
        skip_end = int(n_files * self.skip_ratio)
        train_end = int(n_files * (self.skip_ratio + self.train_ratio))
        
        healthy_files_for_baseline = self.files[skip_end:train_end]
        all_rms = np.array([self.file_rms[f] for f in healthy_files_for_baseline])
        
        if len(all_rms) > 0:
            self.healthy_rms_mean = float(np.mean(all_rms))
            self.healthy_rms_std = float(np.std(all_rms))
            self.healthy_rms_baseline = float(np.percentile(all_rms, 10))
        else:
            self.healthy_rms_mean = 0.01
            self.healthy_rms_std = 0.001
            self.healthy_rms_baseline = 0.01

        self.samples = []
        self.signal_cache = {}
        self._prepare_samples()

    def _load_or_compute_file_rms(self):
        """Load file_rms.json nếu đã có, ngược lại tính và lưu cache."""
        rms_cache_path = os.path.join(self.data_dir, 'file_rms.json')
        if os.path.exists(rms_cache_path):
            with open(rms_cache_path, 'r') as f:
                file_rms = json.load(f)
            # Kiểm tra xem cache có chứa đủ tất cả các file không (tránh lỗi KeyError nếu dataset vừa được cập nhật thêm)
            if all(fname in file_rms for fname in self.files):
                return file_rms
            print(f"file_rms.json thiếu dữ liệu tại {os.path.basename(self.data_dir)}. Đang tính toán lại...")

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
        if n_samples_per_file < window_size:
            n_windows_per_file = 1
        else:
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
            # [FIX] Đánh giá trên toàn bộ chu kỳ sống (Full Life-cycle) để tỷ lệ nhãn không bị thiên lệch
            file_indices = list(range(n_files))

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
        
        # PADDING LOGIC cho các file ngắn hơn yêu cầu (ví dụ: PRONOSTIA 2560 < 5120)
        L = signal.shape[-1]
        if self.lookback + self.horizon > L:
            padding = self.lookback + self.horizon - L
            signal = torch.nn.functional.pad(signal, (0, padding), 'constant', 0)

        end_x = offset + self.lookback
        end_y = end_x + self.horizon

        x_raw = signal[:, offset:end_x]
        y     = signal[:, end_x:end_y]

        stats = self._compute_physical_stats(x_raw)
        x = x_raw.clone()

        current_file_rms = self.smoothed_file_rms[self.files[f_idx]] # Dùng RMS đã làm mịn
        
        if self.label_strategy == '3sigma':
            threshold = self.healthy_rms_mean + 3.0 * self.healthy_rms_std
            label = 1 if current_file_rms > threshold else 0
        elif self.label_strategy == 'manual':
            label = 0
            if self.manual_fault_start is not None:
                bearing_name = os.path.basename(self.data_dir)
                
                # Xác định điểm bắt đầu lỗi cho vòng bi này
                if isinstance(self.manual_fault_start, dict):
                    start_val = self.manual_fault_start.get(bearing_name, None)
                else:
                    start_val = self.manual_fault_start
                
                if start_val is not None:
                    if isinstance(start_val, str):
                        # So sánh tên file (tìm index hoặc so sánh chuỗi trực tiếp)
                        if start_val in self.files:
                            start_idx = self.files.index(start_val)
                            label = 1 if f_idx >= start_idx else 0
                        else:
                            label = 1 if self.files[f_idx] >= start_val else 0
                    elif isinstance(start_val, int):
                        # So sánh chỉ số file
                        label = 1 if f_idx >= start_val else 0
                else:
                    # Cảnh báo và fallback về rms nếu không tìm thấy cấu hình cho vòng bi này
                    threshold = self.healthy_rms_baseline * self.fault_rms_factor
                    label = 1 if current_file_rms > threshold else 0
            else:
                threshold = self.healthy_rms_baseline * self.fault_rms_factor
                label = 1 if current_file_rms > threshold else 0
        else:
            threshold = self.healthy_rms_baseline * self.fault_rms_factor
            label = 1 if current_file_rms > threshold else 0

        if self.normalize:
            mean_x = x.mean(dim=1, keepdim=True)
            std_x  = x.std(dim=1, keepdim=True) + 1e-8
            x = (x - mean_x) / std_x
            y = (y - mean_x) / std_x

        if hasattr(self, 'oc_values_processed') and self.oc_values_processed is not None:
            oc = torch.from_numpy(self.oc_values_processed[f_idx].astype('float32'))
            return x, y, stats, label, oc
        elif hasattr(self, 'static_oc') and self.static_oc is not None:
            oc = torch.from_numpy(self.static_oc)
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

