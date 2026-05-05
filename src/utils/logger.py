import pandas as pd
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, save_dir='results/logs', filename='experiment_logs.csv'):
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        
    def log_experiment(self, config, metrics):
        """
        Log thông tin hyperparameters và kết quả vào chung 1 file CSV.
        """
        log_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": config.get('model_type', 'Unknown'),
            "Lookback": config['data'].get('lookback', '-'),
            "Horizon": config['data'].get('horizon', '-'),
            "BatchSize": config['training'].get('batch_size', '-'),
            "LR": config['training'].get('learning_rate', '-'),
        }
        
        # Gộp tất cả các giá trị (không phải dạng list như CM)
        for k, v in metrics.items():
            if not isinstance(v, list):
                log_data[k] = v
        
        df = pd.DataFrame([log_data])
        
        if os.path.exists(self.filepath):
            df.to_csv(self.filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(self.filepath, index=False)
        
        print(f"Log experiment đã lưu vào {self.filepath}")
