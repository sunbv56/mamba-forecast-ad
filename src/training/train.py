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

from src.data import B02Dataset
from src.models.baselines.lstm import LSTMForecaster
from src.models.baselines.tcn import TCNForecaster
from src.models.baselines.modern_tcn import ModernTCNForecaster
from src.models.baselines.transformer_small import iTransformer, PositionalEncoding
from src.models.baselines.patch_models import PatchTST, PatchLSTM
from src.models.mamba import HybridMambaCNN
from src.evaluation.anomaly_scorer import calculate_anomaly_score
from src.evaluation.metrics import calculate_threshold_3sigma, calculate_metrics

# --- Training and Evaluation Functions ---

def train_one_model(name, model, train_loader, val_loader, test_loader, config, device):
    print(f"\n>>> Training {name}...")
    model.to(device)
    
    lr = float(config['training'].get('learning_rate', 0.001))
    epochs = int(config['training'].get('epochs', 100))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    losses = []
    start_train = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            x, y = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                y_pred = model(x)
                loss = criterion(y_pred, y)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.6f}")
            
    train_duration = time.time() - start_train
    print(f"Training took {train_duration:.2f} seconds")

    # --- Threshold Calculation (from Train Set) ---
    print(f"Calculating threshold for {name}...")
    train_scores_sample = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i > 20: break 
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(x)
            scores = calculate_anomaly_score(y, y_pred, metric='mse')
            train_scores_sample.extend(scores.tolist())
            
    threshold = calculate_threshold_3sigma(np.array(train_scores_sample))
    print(f"Threshold (3-sigma): {threshold:.6f}")

    # --- Evaluation on Test Set ---
    print(f"Evaluating {name} on Test Set...")
    test_scores = []
    latencies = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            
            start_inf = time.time()
            y_pred = model(x)
            latencies.append((time.time() - start_inf) / x.size(0))
            
            scores = calculate_anomaly_score(y, y_pred, metric='mse')
            test_scores.extend(scores.tolist())
    
    test_scores = np.array(test_scores)
    
    # Combined metrics logic from notebook
    combined_scores = np.concatenate([train_scores_sample, test_scores])
    combined_labels = np.concatenate([
        np.zeros(len(train_scores_sample)), 
        np.ones(len(test_scores))
    ]).astype(int)
    
    pred_labels = (combined_scores > threshold).astype(int)
    
    try:
        metrics = calculate_metrics(combined_scores, combined_labels, threshold)
        precision_path, recall_path, _ = precision_recall_curve(combined_labels, combined_scores)
        auprc = auc_score_func(recall_path, precision_path)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics = {'AUC': 0.0}
        auprc = 0.0

    calc_precision = precision_score(combined_labels, pred_labels, zero_division=0)
    calc_recall = recall_score(combined_labels, pred_labels, zero_division=0)
    calc_f1 = f1_score(combined_labels, pred_labels, zero_division=0)
    
    detection_rate = np.sum(test_scores > threshold) / len(test_scores)
    avg_latency = np.mean(latencies) * 1000

    print(f"✅ {name} Finished!")
    print(f"   > F1: {calc_f1:.4f} | AUC: {metrics.get('AUC', 0):.4f} | AUPRC: {auprc:.4f}")
    print(f"   > Precision: {calc_precision:.4f} | Recall: {calc_recall:.4f}")
    print(f"   > Latency: {avg_latency:.4f} ms/sample | Detection Rate: {detection_rate:.2%}")

    # Save results
    result = {
        'name': name,
        'f1': float(calc_f1),
        'auc': float(metrics.get('AUC', 0)),
        'precision': float(calc_precision),
        'recall': float(calc_recall),
        'auprc': float(auprc),
        'latency': float(avg_latency),
        'detection_rate': float(detection_rate),
        'threshold': float(threshold),
        'train_duration': float(train_duration)
    }

    # Save model
    save_dir = config['training'].get('save_dir', 'results/models')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{name.lower().replace('-', '_')}_best.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

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
    parser.add_argument("--subset_ratio", type=int, default=30, help="Take every N-th sample for faster training (default: 1).")
    
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
    processed_dir = config['data']['processed_dir']
    lookback = config['data'].get('lookback', 1024)
    horizon = config['data'].get('forecast_len', 512)
    stride = config['data'].get('stride', 512)
    
    print(f"Loading datasets from {processed_dir}...")
    train_dataset = B02Dataset(processed_dir, lookback, horizon, stride, split='train')
    val_dataset = B02Dataset(processed_dir, lookback, horizon, stride, split='val')
    test_dataset = B02Dataset(processed_dir, lookback, horizon, stride, split='test')
    
    # Subset logic for faster experimentation
    if args.subset_ratio > 1:
        print(f"Applying subset sampling: 1/{args.subset_ratio}")
        train_dataset = Subset(train_dataset, range(0, len(train_dataset), args.subset_ratio))
        val_dataset = Subset(val_dataset, range(0, len(val_dataset), args.subset_ratio // 3 + 1))
        test_dataset = Subset(test_dataset, range(0, len(test_dataset), args.subset_ratio // 3 + 1))
        
    batch_size = int(config['training'].get('batch_size', 128))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # --- Model Initialization ---
    all_models = {
        "LSTM": LSTMForecaster(input_dim=2, hidden_dim=80, num_layers=3, horizon=horizon),
        "PatchLSTM": PatchLSTM(in_channels=2, patch_size=64, stride=64, d_model=80, num_layers=3, horizon=horizon),
        # "TCN": TCNForecaster(input_dim=2, num_channels=[64]*4, kernel_size=3, horizon=horizon),
        "ModernTCN": ModernTCNForecaster(input_dim=2, d_model=96, num_layers=3, kernel_size=17, horizon=horizon),
        "iTransformer": iTransformer(input_dim=2, lookback=lookback, d_model=64, nhead=4, num_layers=3, horizon=horizon),
        "PatchTST": PatchTST(in_channels=2, lookback=lookback, patch_size=64, stride=64, d_model=64, nhead=4, num_layers=3, horizon=horizon),
        "Mamba-Hybrid": HybridMambaCNN({
            'model': {
                'cnn_out_channels': 64, 'mamba_d_model': 64, 'mamba_n_layer': 4,
                'mamba_d_state': 16, 'mamba_d_conv': 4, 'mamba_expand': 2,
                'forecast_len': horizon, 'patch_size': 64, 'stride': 64,
                'in_channels': 2, 'out_channels': 2
            },
            'data': {'patch_size': 64, 'stride': 64}
        })
    }
    
    if args.model == "all":
        models_to_train = all_models
    elif args.model in all_models:
        models_to_train = {args.model: all_models[args.model]}
    else:
        print(f"Error: Model {args.model} not found.")
        sys.exit(1)
        
    results = {}
    for name, model in models_to_train.items():
        res = train_one_model(name, model, train_loader, val_loader, test_loader, config, device)
        results[name] = res
        
    # Final Results Summary
    print("\n" + "="*50)
    print(f"{'Model':<20} | {'F1':<8} | {'AUC':<8} | {'Latency (ms)':<12}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} | {res['f1']:<8.4f} | {res['auc']:<8.4f} | {res['latency']:<12.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
