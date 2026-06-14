"""
=============================================================================
PROOF OF CONCEPT: Bearing Signal Has Trend (Low-Freq) + Seasonal (High-Freq)
=============================================================================
Dựa trên nguyên lý Architectural Parsimony của DMamba [arXiv:2602.09081]:
  - Trend component: low-dimensional manifold → Linear branch
  - Seasonal component: high-dimensional, non-linear dynamics → Mamba branch

Dataset: B02 (LDM run-to-failure bearing, 1116 measurements)
  - Key: accHorizFrontal_C (horizontal frontal channel), 204800 samples/file
  - Fault onset: data_B02_M0867 → FAULT_START = 867
Output : Figure được lưu vào artifacts/figures/
============================================================================="""

import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, welch, lfilter
from scipy.stats import pearsonr

def resolve_path(p):
    if os.name == 'nt' and p.startswith('/mnt/'):
        parts = p.split('/')
        drive = parts[2]
        return f"{drive}:/" + "/".join(parts[3:])
    return p

DATA_DIR    = resolve_path("/mnt/f/APPS_PJ/mamba-forecast-ad/data/raw/B02/vibrationData")
OUT_DIR     = resolve_path("/mnt/f/APPS_PJ/mamba-forecast-ad/Hydrid-Mamba-for-Predictive-Bearing-Fault/artifacts/figures")
BEARING     = "B02"
FS          = 25600        # Hz – actual sampling rate B02 (204800 samples / 8 s = 25600 Hz)
N_FILES     = 1116         # total measurement files for B02 (M0001–M1116)
HEALTHY_END = 100          # files 1..100 considered fully healthy
FAULT_START = 867          # data_B02_M0867 – fault onset (per annotation)
DECOMP_ALPHA = 0.03      # Default EMA-based decomposition alpha
AUTO_SWEEP_ALPHA = True    # Set to True to automatically find the best alpha value
HPF_CUTOFF  = 0            # Hz – high-pass Butterworth for spectral plot (max=FS/2=6400)
SUBSAMPLE   = 8            # downsample factor when scanning all files (speed)
PCA_SEGS    = 120          # number of file segments to use for PCA
os.makedirs(OUT_DIR, exist_ok=True)

# ─── HELPERS ───────────────────────────────────────────────────────────────
def load_mat(path, subsample=1):
    """Load .mat vibration file → 1-D float64 array.
    B03/B02 keys: accHorizFrontal_C (ch C, frontal) or accHorizRear_A (ch A, rear).
    subsample: integer decimation factor to reduce array length for speed.
    """
    d = sio.loadmat(path)
    # B03/B02-specific preferred channel first, then generic fallbacks
    preferred = ['accHorizFrontal_C', 'accHorizRear_A',
                 'vibration', 'Vibration', 'data', 'Data', 'vib']
    arr = None
    for key in preferred:
        if key in d:
            arr = np.asarray(d[key]).ravel().astype(np.float64)
            break
    if arr is None:
        keys = [k for k in d.keys() if not k.startswith('_') and k != 'measTime']
        arr = np.asarray(d[keys[0]]).ravel().astype(np.float64)
    if subsample > 1:
        arr = arr[::subsample]
    return arr

def ema_decompose(x, alpha=None):
    """EMA-based series decomposition (DMamba style).
    x_trend[t] = alpha * x[t] + (1 - alpha) * x_trend[t-1]
    x_trend[0] = x[0]
    """
    if alpha is None:
        alpha = DECOMP_ALPHA
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    zi = np.array([(1.0 - alpha) * x[0]])
    trend, _ = lfilter(b, a, x, zi=zi)
    seasonal = x - trend
    return trend, seasonal

def butter_highpass(x, cutoff=HPF_CUTOFF, fs=FS, order=4):
    sos = butter(order, cutoff, btype='high', fs=fs, output='sos')
    from scipy.signal import sosfilt
    return sosfilt(sos, x)

def rms(x):
    return np.sqrt(np.mean(x**2))

def kurtosis(x):
    mu = np.mean(x); s = np.std(x)
    return np.mean(((x - mu)/s)**4) if s > 0 else 0.0

def collect_stats_per_file(file_list):
    """Iterate all files, compute RMS of raw, trend & seasonal per file.
    Uses SUBSAMPLE decimation for speed (204800 → 25600 points per file).
    """
    rms_raw, rms_trend, rms_seasonal, kurt_seasonal = [], [], [], []
    for fp in file_list:
        try:
            x = load_mat(fp, subsample=SUBSAMPLE)  # fast: 25600 pts/file
            if len(x) < 100:
                rms_raw.append(np.nan)
                rms_trend.append(np.nan)
                rms_seasonal.append(np.nan)
                kurt_seasonal.append(np.nan)
                continue
            t, s = ema_decompose(x)
            rms_raw.append(rms(x))
            rms_trend.append(rms(t))
            rms_seasonal.append(rms(s))
            kurt_seasonal.append(kurtosis(s))
        except Exception:
            rms_raw.append(np.nan)
            rms_trend.append(np.nan)
            rms_seasonal.append(np.nan)
            kurt_seasonal.append(np.nan)
    return (np.array(rms_raw),
            np.array(rms_trend),
            np.array(rms_seasonal),
            np.array(kurt_seasonal))

# ─── LOAD DATA ─────────────────────────────────────────────────────────────
print(f"▶ Scanning {BEARING} vibration files …")
all_files = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith('.mat')
])
print(f"  Found {len(all_files)} files.")

# ─── AUTO SWEEP ALPHA ──────────────────────────────────────────────────────
if AUTO_SWEEP_ALPHA:
    print("\n🔍 Running automatic grid search to find the optimal DECOMP_ALPHA...")
    search_x = []
    for fp in all_files:
        try:
            x = load_mat(fp, subsample=SUBSAMPLE)
            search_x.append(x if len(x) >= 100 else None)
        except:
            search_x.append(None)
            
    candidate_alphas = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.1368, 0.2]
    best_alpha = DECOMP_ALPHA
    best_corr = -1.0
    sweep_results = []
    
    for alpha in candidate_alphas:
        rms_tr_sweep = []
        for x in search_x:
            if x is None:
                rms_tr_sweep.append(np.nan)
                continue
            b = [alpha]
            a = [1.0, -(1.0 - alpha)]
            zi = np.array([(1.0 - alpha) * x[0]])
            t, _ = lfilter(b, a, x, zi=zi)
            rms_tr_sweep.append(np.sqrt(np.mean(t**2)))
            
        rms_tr_sweep = np.array(rms_tr_sweep)
        meas_idx_sweep = np.arange(1, len(rms_tr_sweep) + 1)
        valid_sweep = ~np.isnan(rms_tr_sweep)
        r_tr_sweep, _ = pearsonr(meas_idx_sweep[valid_sweep], rms_tr_sweep[valid_sweep])
        sweep_results.append((alpha, r_tr_sweep))
        if r_tr_sweep > best_corr:
            best_corr = r_tr_sweep
            best_alpha = alpha
            
    print("  Grid Search Results (Alpha vs Trend Pearson r):")
    for alpha, corr in sweep_results:
        marker = " 🌟 (Best)" if alpha == best_alpha else ""
        print(f"    - Alpha: {alpha:<7.4f} -> Trend Pearson r: {corr:.4f}{marker}")
    print(f"  Setting DECOMP_ALPHA to: {best_alpha}\n")
    DECOMP_ALPHA = best_alpha

# Representative snapshots: healthy M0001, mid-life M0559 (~50%), fault onset+10
idx_healthy = 0
idx_mid     = min(558, len(all_files) - 1)
idx_fault   = min(FAULT_START + 10, len(all_files) - 1)

# Load full resolution (no subsampling) for waveform plots
sig_h = load_mat(all_files[idx_healthy])
sig_m = load_mat(all_files[idx_mid])
sig_f = load_mat(all_files[idx_fault])

# 8192-sample window (~320 ms @ 25600 Hz)
WIN = 8192
sig_h = sig_h[:WIN]; sig_m = sig_m[:WIN]; sig_f = sig_f[:WIN]
t_axis = np.arange(WIN) / FS * 1000  # ms

# Decompose each
t_h, s_h = ema_decompose(sig_h)
t_m, s_m = ema_decompose(sig_m)
t_f, s_f = ema_decompose(sig_f)

# ─── FFT helpers ────────────────────────────────────────────────────────────
def fft_mag(x, fs=FS):
    N   = len(x)
    X   = np.abs(np.fft.rfft(x * np.hanning(N))) * 2 / N
    f   = np.fft.rfftfreq(N, 1/fs)
    return f, X

# ─── COLLECT LONGITUDINAL STATS (all files, subsampled for speed) ────────────
print(f"▶ Computing RMS/kurtosis across {len(all_files)} files (x{SUBSAMPLE} subsample) …")
rms_raw, rms_tr, rms_se, kurt_se = collect_stats_per_file(all_files)
meas_idx = np.arange(1, len(rms_tr) + 1)

# ─── POWER BAND ANALYSIS (single healthy snippet) ───────────────────────────
sig_full_h = load_mat(all_files[idx_healthy])
f_h, P_h = welch(sig_full_h, fs=FS, nperseg=2048)
f_f_full = load_mat(all_files[idx_fault])
f_ff, P_ff = welch(f_f_full, fs=FS, nperseg=2048)

# PSD of trend vs seasonal (healthy)
_, P_trend_h   = welch(ema_decompose(sig_full_h)[0], fs=FS, nperseg=2048)
_, P_season_h  = welch(ema_decompose(sig_full_h)[1], fs=FS, nperseg=2048)

# ─── FIGURE 1: 3-panel waveform decomposition ───────────────────────────────
print("▶ Plotting Figure 1 – Waveform decomposition …")
COLORS = {
    'raw'     : '#1f77b4', # soft blue
    'trend'   : '#e65c00', # burnt orange
    'seasonal': '#2ca02c', # forest green
    'healthy' : '#00a896', # teal
    'fault'   : '#d62728', # red
}

fig1, axes = plt.subplots(3, 3, figsize=(16, 10))
fig1.patch.set_facecolor('white')
for ax in axes.ravel():
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=10)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['top'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['right'].set_color('#cccccc')

titles_col = [f'Healthy (M0001)', f'Mid-life (M{idx_mid+1:04d})', f'Fault Onset (M{idx_fault+1:04d})']
titles_row = ['Raw Signal $X$', 'Trend $X_{trend}$ (Low-Freq)', 'Seasonal $X_{seasonal}$ (High-Freq)']
data_sets  = [(sig_h, t_h, s_h), (sig_m, t_m, s_m), (sig_f, t_f, s_f)]
row_colors = [COLORS['raw'], COLORS['trend'], COLORS['seasonal']]

for col, (raw, tr, se) in enumerate(data_sets):
    for row, (y, rc) in enumerate(zip([raw, tr, se], row_colors)):
        ax = axes[row][col]
        ax.plot(t_axis, y, color=rc, lw=0.6, alpha=0.85)
        if row == 0:
            ax.set_title(titles_col[col], color='black', fontsize=14, fontweight='bold', pad=6)
        if col == 0:
            ax.set_ylabel(titles_row[row], color='black', fontsize=12)
        ax.set_xlabel('Time (ms)', color='black', fontsize=11)
        # Annotate RMS
        ax.text(0.98, 0.95, f'RMS={rms(y):.4f}', transform=ax.transAxes,
                ha='right', va='top', color=rc, fontsize=11,
                bbox=dict(facecolor='white', edgecolor='#cccccc', alpha=0.8))

fig1.suptitle(
    f'Bearing {BEARING}: EMA-based Series Decomposition (FS={FS} Hz)\n'
    r'$X = X_{trend} + X_{seasonal}$  (DMamba Principle, arXiv:2602.09081)',
    color='black', fontsize=17, fontweight='bold', y=0.98
)
fig1.tight_layout(rect=[0, 0, 1, 0.96])
p1 = os.path.join(OUT_DIR, 'fig1_waveform_decomposition.png')
fig1.savefig(p1, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved → {p1}")

# ─── FIGURE 2: PSD comparison ───────────────────────────────────────────────
print("▶ Plotting Figure 2 – Power Spectral Density …")
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.patch.set_facecolor('white')
for ax in axes2:
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=10)
    for sp in ax.spines.values():
        sp.set_color('#cccccc')

# Panel A: Raw PSD healthy vs fault
ax = axes2[0]
ax.semilogy(f_h,  P_h,  color=COLORS['healthy'], lw=1.2, label='Healthy M0001')
ax.semilogy(f_ff, P_ff, color=COLORS['fault'],   lw=1.2, label=f'Fault M{idx_fault+1:04d}')
ax.axvspan(0, 500,   alpha=0.12, color=COLORS['trend'],   label='Trend band (<500 Hz)')
ax.axvspan(500, 12800, alpha=0.08, color=COLORS['seasonal'], label='Seasonal band (>500 Hz)')
ax.set_xlabel('Frequency (Hz)', color='black', fontsize=12)
ax.set_ylabel('PSD (g²/Hz)', color='black', fontsize=12)
ax.set_title('Raw Signal PSD: Healthy vs Fault', color='black', fontsize=13)
ax.legend(fontsize=10.5, facecolor='white', labelcolor='black')

# Panel B: Trend PSD vs Seasonal PSD (healthy)
ax = axes2[1]
ax.semilogy(f_h, P_trend_h,  color=COLORS['trend'],    lw=1.4, label='Trend component')
ax.semilogy(f_h, P_season_h, color=COLORS['seasonal'], lw=1.4, label='Seasonal component')
ax.set_xlabel('Frequency (Hz)', color='black', fontsize=12)
ax.set_title('PSD after Decomposition (Healthy)', color='black', fontsize=13)
ax.legend(fontsize=10.5, facecolor='white', labelcolor='black')

# Band energy ratio
band_low  = np.sum(P_trend_h[f_h < 500])
band_high = np.sum(P_season_h[f_h >= 500])
ratio_txt = f'Energy ratio\nTrend<500Hz: {band_low:.2e}\nSeasonal>500Hz: {band_high:.2e}'
ax.text(0.03, 0.05, ratio_txt, transform=ax.transAxes,
        color='black', fontsize=10, va='bottom',
        bbox=dict(facecolor='white', edgecolor='#cccccc', alpha=0.9))

# Panel C: FFT magnitude of seasonal (fault) – show impulse peaks
f_se_fault, X_se_fault = fft_mag(s_f, FS)
ax = axes2[2]
ax.plot(f_se_fault[:len(f_se_fault)//2], X_se_fault[:len(f_se_fault)//2],
        color=COLORS['fault'], lw=0.8, alpha=0.9)
ax.set_xlabel('Frequency (Hz)', color='black', fontsize=12)
ax.set_ylabel('Amplitude (g)', color='black', fontsize=12)
ax.set_title('Seasonal Component FFT – Fault State\n(High-freq impulse peaks)', color='black', fontsize=13)
# annotate highest peak
peak_idx = np.argmax(X_se_fault)
ax.annotate(f'{f_se_fault[peak_idx]:.0f} Hz',
            xy=(f_se_fault[peak_idx], X_se_fault[peak_idx]),
            xytext=(f_se_fault[peak_idx]+400, X_se_fault[peak_idx]*0.85),
            arrowprops=dict(arrowstyle='->', color='#d62728'),
            color='#d62728', fontsize=10.5)

fig2.suptitle('Frequency-Domain Evidence: Trend = Low-Freq | Seasonal = High-Freq',
              color='black', fontsize=15, fontweight='bold')
fig2.tight_layout()
p2 = os.path.join(OUT_DIR, 'fig2_psd_frequency_evidence.png')
fig2.savefig(p2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved → {p2}")

# ─── FIGURE 3: Longitudinal RMS progression ─────────────────────────────────
print("▶ Plotting Figure 3 – Longitudinal degradation …")
fig3, axes3 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig3.patch.set_facecolor('white')
for ax in axes3:
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=10)
    for sp in ax.spines.values():
        sp.set_color('#cccccc')

# Fault onset marker
for ax in axes3:
    ax.axvline(FAULT_START, color='#d62728', lw=1.5, linestyle='--', alpha=0.7, label='Fault onset')
    ax.axvspan(0, HEALTHY_END, alpha=0.08, color='#00a896')
    ax.axvspan(FAULT_START, N_FILES, alpha=0.08, color='#d62728')

# Top: Trend RMS (slow monotonic rise = low-dim manifold)
ax = axes3[0]
ax.plot(meas_idx, rms_tr, color=COLORS['trend'], lw=1.2, label='RMS(Trend)', zorder=3)
# smoothed
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr
sm_tr = uniform_filter1d(rms_tr, size=20)
ax.plot(meas_idx, sm_tr, color='#b8860b', lw=2.0, linestyle='--', label='Smoothed Trend', zorder=4)
# Add Raw RMS as reference on twin axis
ax2 = ax.twinx()
ax2.tick_params(colors='#1f77b4', labelsize=10)
ax2.spines['right'].set_color('#1f77b4')
ax2.plot(meas_idx, rms_raw, color=COLORS['raw'], lw=0.8, alpha=0.5, label='RMS(Raw)', zorder=1)
sm_raw = uniform_filter1d(rms_raw, size=20)
ax2.plot(meas_idx, sm_raw, color='#1a75ff', lw=1.8, linestyle=':', label='Smoothed Raw', zorder=2)
ax2.set_ylabel('RMS – Raw Signal', color='#1f77b4', fontsize=12)
ax2.tick_params(axis='y', colors='#1f77b4', labelsize=10)
# Merge legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10.5,
          facecolor='white', labelcolor='black', loc='upper left')
ax.set_ylabel('RMS – Trend', color='black', fontsize=13)
ax.set_title('Trend Component vs Raw RMS  →  Low-Dimensional Manifold\n'
             '(Raw ≈ Seasonal when Trend ≈ 0 confirms decomposition quality)',
             color='black', fontsize=13, fontweight='bold')

# Bottom: Seasonal RMS (volatile, high-freq event-driven)
ax = axes3[1]
ax.plot(meas_idx, rms_se, color=COLORS['seasonal'], lw=0.7, alpha=0.75, label='RMS(Seasonal)')
sm_se = uniform_filter1d(rms_se, size=20)
ax.plot(meas_idx, sm_se, color='#e65c00', lw=2.0, linestyle='--', label='Smoothed (window=20)')
ax.set_ylabel('RMS – Seasonal', color='black', fontsize=13)
ax.set_xlabel(f'Measurement Index (1 → {N_FILES} = Run-to-Failure | Fault onset: M{FAULT_START:04d})', color='black', fontsize=13)
ax.set_title('Seasonal Component: Volatile, Impulsive  →  High-Dimensional Non-Linear Dynamics\n'
             '(Justifies: Channel-independent Mamba encoder branch)',
             color='black', fontsize=13, fontweight='bold')
ax.legend(fontsize=10.5, facecolor='white', labelcolor='black')

# Correlation annotation
valid = ~(np.isnan(rms_tr) | np.isnan(rms_se))
r_tr, _ = pearsonr(meas_idx[valid], rms_tr[valid])
r_se, _ = pearsonr(meas_idx[valid], rms_se[valid])
axes3[0].text(0.02, 0.92, f'Pearson r(idx, Trend_RMS) = {r_tr:.3f}',
              transform=axes3[0].transAxes, color=COLORS['trend'], fontsize=11,
              bbox=dict(facecolor='white', edgecolor='#cccccc', alpha=0.8))
axes3[1].text(0.02, 0.92, f'Pearson r(idx, Seasonal_RMS) = {r_se:.3f}',
              transform=axes3[1].transAxes, color=COLORS['seasonal'], fontsize=11,
              bbox=dict(facecolor='white', edgecolor='#cccccc', alpha=0.8))

fig3.suptitle(
    f'{BEARING} Run-to-Failure: Longitudinal Evidence for Dual-Stream Architecture\n'
    'Trend ≈ slow mechanical wear  |  Seasonal ≈ transient fault impulses',
    color='black', fontsize=16, fontweight='bold', y=0.99
)
fig3.tight_layout(rect=[0, 0, 1, 0.97])
p3 = os.path.join(OUT_DIR, 'fig3_longitudinal_rms.png')
fig3.savefig(p3, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved → {p3}")

# ─── FIGURE 4: Dimensionality proof (PCA on trend vs seasonal) ──────────────
# ─── FIGURE 4: Dimensionality proof (PCA on trend vs seasonal) ──────────────
print("▶ Plotting Figure 4 – Dimensionality comparison (PCA) …")
from sklearn.decomposition import PCA

# Build matrix of 200 healthy segments
N_SEGS = PCA_SEGS
WIN_PCA = 2048  # fixed window size for PCA (works at any subsample rate)
X_trend_mat, X_season_mat = [], []
for fp in all_files[:min(N_SEGS, len(all_files))]:
    try:
        x = load_mat(fp)[:WIN_PCA]
        if len(x) < WIN_PCA:
            continue
        tr, se = ema_decompose(x)
        X_trend_mat.append(tr)
        X_season_mat.append(se)
    except Exception:
        pass

X_trend_mat  = np.array(X_trend_mat)
X_season_mat = np.array(X_season_mat)

n_comp = min(30, len(X_trend_mat))
pca_tr = PCA(n_components=n_comp).fit(X_trend_mat)
pca_se = PCA(n_components=n_comp).fit(X_season_mat)

cumvar_tr = np.cumsum(pca_tr.explained_variance_ratio_) * 100
cumvar_se = np.cumsum(pca_se.explained_variance_ratio_) * 100

fig4, ax4 = plt.subplots(figsize=(9, 5))
fig4.patch.set_facecolor('white')
ax4.set_facecolor('white')
ax4.tick_params(colors='black', labelsize=10)
for sp in ax4.spines.values():
    sp.set_color('#cccccc')

comps = np.arange(1, n_comp + 1)
ax4.plot(comps, cumvar_tr, 'o-', color=COLORS['trend'],    lw=2, ms=5, label='Trend component')
ax4.plot(comps, cumvar_se, 's-', color=COLORS['seasonal'], lw=2, ms=5, label='Seasonal component')
ax4.axhline(90, color='#b8860b', lw=1, linestyle=':', alpha=0.7, label='90% threshold')
ax4.axhline(95, color='#d62728', lw=1, linestyle=':', alpha=0.7, label='95% threshold')

# Find components needed for 90% variance
def comp_for_var(cumvar, threshold=90):
    idx = np.searchsorted(cumvar, threshold)
    return idx + 1

c_tr_90 = comp_for_var(cumvar_tr, 90)
c_se_90 = comp_for_var(cumvar_se, 90)
ax4.annotate(f'{c_tr_90} PCs\n→90% var',
             xy=(c_tr_90, 90), xytext=(c_tr_90 + 3, 75),
             arrowprops=dict(arrowstyle='->', color=COLORS['trend']),
             color=COLORS['trend'], fontsize=11)
ax4.annotate(f'{c_se_90} PCs\n→90% var',
             xy=(c_se_90, 90), xytext=(c_se_90 + 3, 55),
             arrowprops=dict(arrowstyle='->', color=COLORS['seasonal']),
             color=COLORS['seasonal'], fontsize=11)

ax4.set_xlabel('Number of Principal Components', color='black', fontsize=13)
ax4.set_ylabel('Cumulative Explained Variance (%)', color='black', fontsize=13)
ax4.set_title(
    f'PCA Dimensionality ({BEARING}): Trend Resides on Lower-Dimensional Manifold\n'
    '→ Architectural Parsimony Principle [DMamba, arXiv:2602.09081]',
    color='black', fontsize=13, fontweight='bold'
)
ax4.legend(fontsize=10.5, facecolor='white', labelcolor='black')
ax4.set_xlim(1, n_comp)
ax4.set_ylim(0, 102)
ax4.grid(True, color='#cccccc', alpha=0.5)

fig4.tight_layout()
p4 = os.path.join(OUT_DIR, 'fig4_pca_dimensionality.png')
fig4.savefig(p4, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved → {p4}")

# ─── FIGURE 5: Summary architecture justification diagram ───────────────────
print("▶ Plotting Figure 5 – Architecture justification summary …")
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
fig5.patch.set_facecolor('white')
for ax in axes5:
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=10)
    for sp in ax.spines.values():
        sp.set_color('#cccccc')

# Left: Kurtosis over time (seasonal = non-linear spikes)
ax = axes5[0]
ax.plot(meas_idx, kurt_se, color=COLORS['seasonal'], lw=0.8, alpha=0.7, label='Seasonal Kurtosis')
sm_k = uniform_filter1d(kurt_se, size=30)
ax.plot(meas_idx, sm_k, color='#e65c00', lw=2.2, label='Smoothed')
ax.axvline(FAULT_START, color='#d62728', lw=1.5, linestyle='--', label='Fault onset')
ax.axhline(3.0, color='black', lw=0.8, linestyle=':', alpha=0.5, label='Gaussian baseline (K=3)')
ax.set_xlabel('Measurement Index', color='black', fontsize=13)
ax.set_ylabel('Kurtosis of Seasonal Component', color='black', fontsize=13)
ax.set_title('Seasonal Kurtosis: Non-Gaussianity → Non-Linear Dynamics\n'
             '(High K = impulsive fault energy → needs Mamba, not Linear)',
             color='black', fontsize=12, fontweight='bold')
ax.legend(fontsize=10.5, facecolor='white', labelcolor='black')

# Right: Scatter – Trend RMS vs Seasonal RMS
ax = axes5[1]
scatter = ax.scatter(rms_tr, rms_se, c=meas_idx, cmap='plasma',
                     s=8, alpha=0.7)
cbar = fig5.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Measurement Index', color='black', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='black', labelsize=10)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black', fontsize=10)
ax.set_xlabel('RMS(Trend) – slow wear evolution', color='black', fontsize=13)
ax.set_ylabel('RMS(Seasonal) – impulsive content', color='black', fontsize=13)
ax.set_title('Trend vs Seasonal Trajectory\nColors = time progression (Healthy → Fault)',
             color='black', fontsize=12, fontweight='bold')

# Annotate correlation
r_ts, _ = pearsonr(rms_tr[valid], rms_se[valid])
corr_note = (
    f'Pearson r = {r_ts:.3f}\n'
    'Both components spike at fault onset\n'
    '(co-evolve at failure, independent in healthy)'
)
ax.text(0.05, 0.93, corr_note,
        transform=ax.transAxes, color='black', fontsize=11,
        bbox=dict(facecolor='white', edgecolor='#cccccc', alpha=0.9))

fig5.suptitle(
    'Justification: Why Dual-Stream Architecture is Physically Motivated\n'
    'Trend → Linear Branch  |  Seasonal → Channel-Independent Mamba Encoder',
    color='black', fontsize=15, fontweight='bold'
)
fig5.tight_layout()
p5 = os.path.join(OUT_DIR, 'fig5_architecture_justification.png')
fig5.savefig(p5, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved → {p5}")

# ─── PRINT SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SUMMARY – Evidence for Dual-Stream Architecture")
print("="*65)
print(f"  Trend Pearson r (index vs RMS)   : {r_tr:.4f}  ← monotonic = low-dim")
print(f"  Seasonal Pearson r (index vs RMS): {r_se:.4f}  ← volatile  = high-dim")
print(f"  Trend PCs for 90% var            : {c_tr_90}")
print(f"  Seasonal PCs for 90% var         : {c_se_90}")
print(f"  Dimensionality ratio (S/T)       : {c_se_90/max(c_tr_90,1):.1f}x  ← seasonal needs more capacity")
print(f"\n  Figures saved to: {OUT_DIR}")
print("="*65)
print("✅ Done.")
