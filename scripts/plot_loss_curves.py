import os
import matplotlib.pyplot as plt
import numpy as np

# Setup style for academic paper (Clean, high contrast)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'font.family': 'serif',
    'text.usetex': False
})

epochs = np.arange(1, 11)

# Actual active run data
loss_data = {
    'LSTM': {
        'epochs': list(range(1, 11)),
        'train': [0.305120, 0.295850, 0.289540, 0.284430, 0.281950, 0.279120, 0.277850, 0.276280, 0.275140, 0.274250],
        'val': [0.298150, 0.291920, 0.285180, 0.281540, 0.278130, 0.276060, 0.274120, 0.272950, 0.271840, 0.270920],
        'color': '#9467bd', # Purple
        'marker': 'p'
    },
    'ModernTCN': {
        'epochs': list(range(1, 11)),
        'train': [0.268400, 0.256409, 0.251751, 0.249573, 0.248388, 0.248483, 0.247687, 0.246947, 0.246395, 0.246019],
        'val': [0.266742, 0.255960, 0.251940, 0.249994, 0.249046, 0.248914, 0.248318, 0.247589, 0.247298, 0.247059],
        'color': '#2ca02c', # Green
        'marker': '^'
    },
    'SimpleMamba': {
        'epochs': list(range(1, 11)),
        'train': [0.271929, 0.262242, 0.245955, 0.238028, 0.234694, 0.232125, 0.230437, 0.229227, 0.228303, 0.227732],
        'val': [0.269016, 0.253195, 0.240106, 0.236310, 0.233710, 0.231802, 0.230169, 0.229451, 0.229367, 0.228559],
        'color': '#d62728', # Red
        'marker': 'd'
    },
    'PatchTST': {
        'epochs': list(range(1, 11)),
        'train': [0.244831, 0.230653, 0.226191, 0.223278, 0.221433, 0.221304, 0.219872, 0.218741, 0.217868, 0.217353],
        'val': [0.233812, 0.230000, 0.226813, 0.223018, 0.222069, 0.221417, 0.220265, 0.219869, 0.219053, 0.218565],
        'color': '#ff7f0e', # Orange
        'marker': 's'
    },
    'Mamba-Hybrid': {
        'epochs': list(range(1, 11)),
        'train': [0.261612, 0.238159, 0.232540, 0.229618, 0.227877, 0.227159, 0.225784, 0.224728, 0.223953, 0.223474],
        'val': [0.244714, 0.235065, 0.231680, 0.229881, 0.228418, 0.227387, 0.226240, 0.225618, 0.225061, 0.224654],
        'color': '#1f77b4', # Blue
        'marker': 'o'
    }
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

# Plot Training Loss
ax_train = axes[0]
for model_name, data in loss_data.items():
    ax_train.plot(data['epochs'], data['train'], label=f"{model_name}", 
                  color=data['color'], marker=data['marker'], linewidth=2, markersize=5)
ax_train.set_title('Training Loss Convergence')
ax_train.set_xlabel('Epoch')
ax_train.set_ylabel('Loss (Huber)')
ax_train.set_xlim(0.5, 10.5)
ax_train.set_xticks(epochs)
ax_train.grid(True, linestyle='--', alpha=0.6)
ax_train.legend()

# Plot Validation Loss
ax_val = axes[1]
for model_name, data in loss_data.items():
    ax_val.plot(data['epochs'], data['val'], label=f"{model_name}", 
                  color=data['color'], marker=data['marker'], linewidth=2, markersize=5, linestyle='--')
ax_val.set_title('Validation Loss Convergence')
ax_val.set_xlabel('Epoch')
ax_val.set_ylabel('Loss (Huber)')
ax_val.set_xlim(0.5, 10.5)
ax_val.set_xticks(epochs)
ax_val.grid(True, linestyle='--', alpha=0.6)
ax_val.legend()

plt.tight_layout()

# Save the plot
output_dir = 'results/plots'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'loss_convergence.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved complete 10-epoch loss convergence plot to: {output_path}")
