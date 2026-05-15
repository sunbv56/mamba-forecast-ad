import os
import gc
import sys
import time
import yaml
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc as auc_score_func
from sklearn.metrics import precision_score, recall_score, f1_score

# Add src to sys.path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data import BearingDataset, MultiBearingDataset
from src.models.baselines.lstm import LSTMForecaster
from src.models.baselines.tcn import TCNForecaster
from src.models.baselines.modern_tcn import ModernTCNForecaster
from src.models.baselines.transformer_small import iTransformer, PositionalEncoding
from src.models.baselines.patch_models import PatchTST, PatchLSTM
from src.models.mamba import HybridMambaCNN, MambaTS, MambaTSOfficial, MambaTSConfig
from src.evaluation.anomaly_scorer import calculate_anomaly_score

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
from src.evaluation.metrics import calculate_threshold_3sigma, calculate_threshold_robust, calculate_threshold_percentile, calculate_threshold_gmm, find_best_threshold, calculate_metrics, calculate_threshold_pot

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- Training and Evaluation Functions ---

def train_one_model(name, model, train_loader, val_loader, test_loader, config, device, config_name="default"):
    print(f"\n>>> Training {name} (Config: {config_name})...")
    model.to(device)
    
    lr = float(config['training'].get('learning_rate', 0.001))
    epochs = int(config['training'].get('epochs', 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # HuberLoss: robust hơn MSELoss với tín hiệu rung động có outlier cao
    criterion = nn.HuberLoss(delta=1.0)
    # CosineAnnealingLR: giảm LR từ từ theo hình cos, giúp Mamba thoát local minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Early Stopping setup
    save_dir = config['training'].get('save_dir', 'results/models')
    os.makedirs(save_dir, exist_ok=True)
    
    # [NEW] Include config name in filename
    model_slug = name.lower().replace('-', '_')
    best_model_path = os.path.join(save_dir, f"{model_slug}_{config_name}_best.pth")
    early_stopping = EarlyStopping(patience=3, verbose=True, path=best_model_path)
    
    losses = []
    start_train = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            x, y = batch[0].to(device), batch[1].to(device)
            stats = batch[2].to(device) if len(batch) > 2 and batch[2].shape[-1] == 8 else None
            oc = batch[4].to(device) if len(batch) > 4 else None
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                if stats is not None and isinstance(model, HybridMambaCNN):
                    y_pred = model(x, stats)
                elif oc is not None and isinstance(model, MambaTSOfficial):
                    y_pred = model(x, oc)
                else:
                    y_pred = model(x)
                loss = criterion(y_pred, y)
            
            if scaler:
                scaler.scale(loss).backward()
                
                # [NEW] Update VAST state for MambaTS-Official
                if isinstance(model, MambaTSOfficial) and model.configs.VPT_mode == 1:
                    with torch.no_grad():
                        sample_loss = torch.mean((y_pred - y)**2, dim=(1, 2)).detach()
                        model.batch_update_state(sample_loss)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # [NEW] Update VAST state for MambaTS-Official
                if isinstance(model, MambaTSOfficial) and model.configs.VPT_mode == 1:
                    with torch.no_grad():
                        sample_loss = torch.mean((y_pred - y)**2, dim=(1, 2)).detach()
                        model.batch_update_state(sample_loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.6f}")
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                stats = batch[2].to(device) if len(batch) > 2 and batch[2].shape[-1] == 8 else None
                oc = batch[4].to(device) if len(batch) > 4 else None
                
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    if stats is not None and isinstance(model, HybridMambaCNN):
                        y_pred = model(x, stats)
                    elif oc is not None and isinstance(model, MambaTSOfficial):
                        y_pred = model(x, oc)
                    else:
                        y_pred = model(x)
                    loss = criterion(y_pred, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch}/{epochs}] - Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        
        scheduler.step()
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
            
    train_duration = time.time() - start_train
    print(f"Training took {train_duration:.2f} seconds")

    # Load the best model weights
    print(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # --- [FIX #3] Threshold Calculation (from ALL Train batches — no 21-batch limit) ---
    print(f"Calculating threshold for {name}...")
    train_scores_sample = []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            stats = batch[2].to(device) if len(batch) > 2 and batch[2].shape[-1] == 8 else None
            oc = batch[4].to(device) if len(batch) > 4 else None
            
            if stats is not None and isinstance(model, HybridMambaCNN):
                y_pred = model(x, stats)
            elif oc is not None and isinstance(model, MambaTSOfficial):
                y_pred = model(x, oc)
            else:
                y_pred = model(x)
            scores = calculate_anomaly_score(y, y_pred, metric='mse', normalized=False)
            train_scores_sample.extend(scores.tolist())

    train_scores_sample = np.array(train_scores_sample)
    threshold_3s  = calculate_threshold_3sigma(train_scores_sample)
    threshold_rb  = calculate_threshold_robust(train_scores_sample)
    threshold_pc  = calculate_threshold_percentile(train_scores_sample, q=99.7)
    threshold_pot = calculate_threshold_pot(train_scores_sample, q=1e-3)

    # --- [FIX #1] Evaluation on Test Set — thu thập actual labels từ dataset ---
    print(f"Evaluating {name} on Test Set...")
    test_scores   = []
    test_labels   = []   # [FIX #1] actual window-level labels
    test_latencies = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            stats = batch[2].to(device) if len(batch) > 2 and batch[2].shape[-1] == 8 else None
            oc = batch[4].to(device) if len(batch) > 4 else None

            start_inf = time.time()
            if stats is not None and isinstance(model, HybridMambaCNN):
                y_pred = model(x, stats)
            elif oc is not None and isinstance(model, MambaTSOfficial):
                y_pred = model(x, oc)
            else:
                y_pred = model(x)
            test_latencies.append((time.time() - start_inf) / x.size(0))

            scores = calculate_anomaly_score(y, y_pred, metric='mse', normalized=False)
            test_scores.extend(scores.tolist())

            # batch[3] = window-level label (int 0/1) từ BearingDataset
            if len(batch) > 3:
                labels_batch = batch[3]
                if hasattr(labels_batch, 'numpy'):
                    test_labels.extend(labels_batch.numpy().tolist())
                else:
                    test_labels.extend([int(l) for l in labels_batch])
            else:
                test_labels.extend([0] * x.size(0))  # fallback

    test_scores      = np.array(test_scores)
    test_labels      = np.array(test_labels, dtype=int)
    test_latency_avg = np.mean(test_latencies) * 1000

    n_fault = int(test_labels.sum())
    n_total = len(test_labels)
    print(f"  Test label distribution: {n_fault}/{n_total} anomaly windows ({n_fault/n_total:.1%})")

    # GMM cần cả healthy và anomaly scores để fit 2 cụm
    combined_for_gmm = np.concatenate([train_scores_sample, test_scores])
    threshold_gmm = calculate_threshold_gmm(combined_for_gmm)

    # [FIX #1] Optimal threshold: tìm trên test set với actual labels
    threshold_opt, _ = find_best_threshold(test_scores, test_labels)

    # --- 2. Evaluation on 4000 Random Train Samples (Healthy Check) ---
    print(f"\n>>> EVALUATION ON TRAIN (4000 random healthy samples)...")
    random_train_scores = []
    train_inf_latencies = []
    with torch.no_grad():
        indices = np.random.choice(len(train_loader.dataset), min(4000, len(train_loader.dataset)), replace=False)
        temp_subset = Subset(train_loader.dataset, indices)
        temp_loader = DataLoader(temp_subset, batch_size=config['training'].get('batch_size', 128), shuffle=False)
        
        for batch in temp_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            stats = batch[2].to(device) if len(batch) > 2 and batch[2].shape[-1] == 8 else None
            oc = batch[4].to(device) if len(batch) > 4 else None
            
            start_inf = time.time()
            if stats is not None and isinstance(model, HybridMambaCNN):
                y_pred = model(x, stats)
            elif oc is not None and isinstance(model, MambaTSOfficial):
                y_pred = model(x, oc)
            else:
                y_pred = model(x)
            train_inf_latencies.append((time.time() - start_inf) / x.size(0))
            scores = calculate_anomaly_score(y, y_pred, metric='mse', normalized=False)
            random_train_scores.extend(scores.tolist())
    
    rt_scores = np.array(random_train_scores)
    rt_labels = np.zeros(len(rt_scores)).astype(int)
    train_latency_avg = np.mean(train_inf_latencies) * 1000
    
    for t_name, t_val in [("3-Sigma", threshold_3s), ("Robust", threshold_rb), ("Percentile", threshold_pc), ("POT", threshold_pot), ("Self-Learn", threshold_gmm), ("Optimal", threshold_opt)]:
        m = calculate_metrics(rt_scores, rt_labels, t_val)
        det_rate = np.sum(rt_scores > t_val) / len(rt_scores)
        print(f"   [{t_name:<10}] > F1: {m.get('F1', 0):.4f} | FAR: {m.get('FAR', 0):.4f} | Det Rate: {det_rate:.2%} | Thresh: {t_val:.6f}")
    print(f"   > Avg Latency: {train_latency_avg:.4f} ms/sample")

    # --- [FIX #1] Evaluation on Test Set — dùng actual labels ---
    print(f"\n>>> EVALUATION ON TEST (Anomaly Detection Performance)...")
    if len(np.unique(test_labels)) > 1:
        precision_path, recall_path, _ = precision_recall_curve(test_labels, test_scores)
        auprc = auc_score_func(recall_path, precision_path)
    else:
        auprc = 0.0
        print("  [WARN] Test set contains only one class — AUPRC undefined.")

    for t_name, t_val in [("3-Sigma", threshold_3s), ("Robust", threshold_rb), ("Percentile", threshold_pc),
                          ("POT", threshold_pot), ("Self-Learn", threshold_gmm), ("Optimal", threshold_opt)]:
        m = calculate_metrics(test_scores, test_labels, t_val)
        det_rate = np.sum(test_scores > t_val) / len(test_scores)
        print(f"   [{t_name:<10}] > F1: {m.get('F1', 0):.4f} | FAR: {m.get('FAR', 0):.4f} | Det Rate: {det_rate:.2%} | Thresh: {t_val:.6f}")
    print(f"   > AUPRC: {auprc:.4f} | Avg Latency: {test_latency_avg:.4f} ms/sample")

    print(f"\n✅ {name} Finished!")

    # Save results (using Robust for summary) — actual labels
    m_rb_final  = calculate_metrics(test_scores, test_labels, threshold_rb)
    m_3s_final  = calculate_metrics(test_scores, test_labels, threshold_3s)
    m_pot_final = calculate_metrics(test_scores, test_labels, threshold_pot)
    
    result = {
        'name': name,
        'f1_3s': float(m_3s_final.get('F1', 0)),
        'f1_rb': float(m_rb_final.get('F1', 0)),
        'f1_pot': float(m_pot_final.get('F1', 0)),
        'auc': float(m_rb_final.get('AUC', 0)),
        'far_rb': float(m_rb_final.get('FAR', 0)),
        'auprc': float(auprc),
        'latency': float(test_latency_avg),
        'train_duration': float(train_duration)
    }

    # Best model is already saved by EarlyStopping, so we don't need to save it again here
    # unless we want to ensure the final result dict uses the correct path.
    model_path = best_model_path
    print(f"Best model available at {model_path}")

    # Cleanup memory
    del model
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return result

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate forecasting models for anomaly detection.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--model", type=str, default="all", help="Model name to train (all, LSTM, PatchLSTM, TCN, ModernTCN, Transformer, PatchTransformer, Mamba-Hybrid)")
    parser.add_argument("--epochs", type=int, help="Override epochs from config.")
    parser.add_argument("--batch_size", type=int, help="Override batch size.")
    parser.add_argument("--data_dir", type=str, help="Override data directory.")
    parser.add_argument("--file_subset_ratio", type=int, default=1,
                        help="[FIX #3] Sample every N-th FILE in train+val splits (preserves temporal continuity). Default: 10.")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Overrides
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.data_dir: config['data']['processed_dir'] = args.data_dir
    
    device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # --- Dataset Setup ---
    # Ưu tiên train_datasets/test_datasets nếu có, ngược lại dùng processed_dir
    train_dirs = config['data'].get('train_datasets')
    test_dirs = config['data'].get('test_datasets')
    
    if train_dirs is None:
        train_dirs = [config['data']['processed_dir']]
    if test_dirs is None:
        test_dirs = [config['data']['processed_dir']]

    window_stride = config['data'].get('window_stride', 1024)
    lookback = config['data'].get('lookback', 4096)
    horizon = config['data'].get('horizon', 1024)
    sampling_rate = config['data'].get('sampling_rate', 128000)
    
    # Model patching params
    patch_size = config['model'].get('patch_size', 64)
    patch_stride = config['model'].get('patch_stride', 32)
    trend_downsample = config['model'].get('trend_downsample', 1)
    
    train_ratio = config['data'].get('train_ratio', 0.5)
    skip_ratio = config['data'].get('skip_ratio', 0.1)
    highpass_freq = config['data'].get('highpass_freq', 1000)
    
    print(f"Loading train datasets from {train_dirs}...")
    train_dataset = MultiBearingDataset(train_dirs, lookback=lookback, horizon=horizon, stride=window_stride, split='train',
                                         file_sample_ratio=args.file_subset_ratio, train_ratio=train_ratio, skip_ratio=skip_ratio, 
                                         normalize=False, highpass_freq=highpass_freq, sampling_rate=sampling_rate)
    
    oc_stats = train_dataset.oc_stats
    
    print(f"Loading val datasets from {train_dirs}...")
    val_dataset   = MultiBearingDataset(train_dirs, lookback=lookback, horizon=horizon, stride=window_stride, split='val',
                                         file_sample_ratio=args.file_subset_ratio, oc_stats=oc_stats, train_ratio=train_ratio, skip_ratio=skip_ratio, 
                                         normalize=False, highpass_freq=highpass_freq, sampling_rate=sampling_rate)
    
    print(f"Loading test datasets from {test_dirs}...")
    test_dataset  = MultiBearingDataset(test_dirs, lookback=lookback, horizon=horizon, stride=window_stride, split='test',
                                         file_sample_ratio=1, oc_stats=oc_stats, train_ratio=train_ratio, skip_ratio=skip_ratio, 
                                         normalize=False, highpass_freq=highpass_freq, sampling_rate=sampling_rate)
        
    batch_size = int(config['training'].get('batch_size', 128))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # --- Model Initialization ---
    all_models = {
        # "LSTM": LSTMForecaster(input_dim=2, hidden_dim=80, num_layers=3, horizon=horizon),
        # "PatchLSTM": PatchLSTM(in_channels=2, patch_size=64, stride=64, d_model=80, num_layers=3, horizon=horizon),
        # "TCN": TCNForecaster(input_dim=2, num_channels=[64]*4, kernel_size=3, horizon=horizon),
        # "ModernTCN": ModernTCNForecaster(input_dim=2, d_model=96, num_layers=3, kernel_size=17, horizon=horizon),
        # "iTransformer": iTransformer(input_dim=2, lookback=lookback, d_model=64, nhead=4, num_layers=3, horizon=horizon),
        # "PatchTST": PatchTST(in_channels=2, lookback=lookback, patch_size=64, stride=64, d_model=64, nhead=4, num_layers=3, horizon=horizon),
        "Mamba1-Hybrid": HybridMambaCNN({
            'model': {
                'mamba_version': 1,
                'mamba_d_model': config['model'].get('mamba_d_model', 64), 
                'mamba_n_layer': config['model'].get('mamba_n_layer', 4),
                'mamba_d_state': config['model'].get('mamba_d_state', 16), 
                'mamba_d_conv': config['model'].get('mamba_d_conv', 4), 
                'mamba_expand': config['model'].get('mamba_expand', 2),
                'forecast_len': horizon, 
                'patch_size': patch_size, 
                'stride': patch_stride,
                'trend_downsample': trend_downsample,
                'in_channels': 2, 'lookback': lookback,
                'decomp_kernel': config['model'].get('decomp_kernel', 25), 
                'use_multiscale': True,
            },
            'data': {
                'patch_size': patch_size, 
                'stride': patch_stride, 
                'lookback': lookback
            }
        }),
        # "MambaTS-Paper": MambaTS(
        #     in_channels=2,
        #     lookback=lookback,
        #     forecast_len=horizon,
        #     patch_size=64,
        #     stride=32,
        #     d_model=32,
        #     n_layers=4,
        #     use_vas=False
        # ),
        "MambaTS-Official": MambaTSOfficial(MambaTSConfig(
            in_channels=2,
            lookback=lookback,
            forecast_len=horizon,
            patch_size=patch_size,
            stride=patch_stride,
            d_model=config['model'].get('mamba_d_model', 64),
            n_layers=config['model'].get('mamba_n_layer', 4),
            dropout=0.2, # Paper suggested 0.2-0.3
            VPT_mode=1, # Enable Variable-Aware Scanning
            ATSP_solver='SA'
        )),
    }
    
    if args.model == "all":
        models_to_train = all_models
    elif args.model in all_models:
        models_to_train = {args.model: all_models[args.model]}
    else:
        print(f"Error: Model {args.model} not found.")
        sys.exit(1)
        
    # Extract config name from filename (e.g., 'configs/small.yaml' -> 'small')
    config_name = os.path.basename(args.config).replace('.yaml', '')
    
    results = {}
    for name, model in models_to_train.items():
        params = count_parameters(model)
        print(f"\n>>> Training {name} (Params: {params:,}) with Config: {config_name}...")
        res = train_one_model(name, model, train_loader, val_loader, test_loader, config, device, config_name=config_name)
        results[name] = res
        
    # Final Results Summary
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'F1 (3s)':<8} | {'F1 (rb)':<8} | {'F1 (POT)':<8} | {'AUC':<8} | {'FAR (rb)':<8}")
    print("-" * 75)
    for name, res in results.items():
        print(f"{name:<20} | {res['f1_3s']:<8.4f} | {res['f1_rb']:<8.4f} | {res.get('f1_pot', 0):<8.4f} | {res['auc']:<8.4f} | {res['far_rb']:<8.4f}")
    print("="*75)

if __name__ == "__main__":
    main()
