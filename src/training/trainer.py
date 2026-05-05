import torch
import torch.nn as nn
from tqdm import tqdm
import os
from src.evaluation.anomaly_scorer import calculate_anomaly_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        
        self.save_dir = config['training'].get('save_dir', 'results/models')
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
                
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
        
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                total_loss += loss.item()
        return total_loss / max(1, len(self.val_loader))

    def train(self):
        epochs = self.config['training']['epochs']
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
                
        return best_val_loss
