#!/usr/bin/env python3
"""
Precompute per-file RMS values for the B02 dataset.
Chạy một lần trước khi training để tăng tốc dataset initialization.

Usage:
    python scripts/precompute_rms.py
    python scripts/precompute_rms.py --data_dir data/processed
"""

import os
import sys
import json
import argparse
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch


def main():
    parser = argparse.ArgumentParser(description="Precompute per-file RMS for B02 dataset.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--train_rms_pct", type=int, default=40, help="Percentile for train/val boundary")
    parser.add_argument("--test_rms_pct",  type=int, default=80, help="Percentile for val/test boundary")
    parser.add_argument("--fault_factor",  type=float, default=3.0, help="fault_rms_factor for window labels")
    args = parser.parse_args()

    data_dir = args.data_dir
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
    print(f"Found {len(files)} .pt files in '{data_dir}'")

    # --- Compute RMS per file ---
    rms_cache_path = os.path.join(data_dir, 'file_rms.json')
    if os.path.exists(rms_cache_path):
        print(f"Cache already exists at {rms_cache_path}. Loading...")
        with open(rms_cache_path, 'r') as f:
            file_rms = json.load(f)
    else:
        file_rms = {}
        for i, fname in enumerate(files):
            fpath = os.path.join(data_dir, fname)
            data = torch.load(fpath, weights_only=True)
            rms = float(data.pow(2).mean().sqrt().item())
            file_rms[fname] = rms
            if (i + 1) % 100 == 0 or (i + 1) == len(files):
                print(f"  [{i+1:4d}/{len(files)}] {fname}  RMS={rms:.4f}")

        with open(rms_cache_path, 'w') as f:
            json.dump(file_rms, f, indent=2)
        print(f"\nSaved RMS cache → {rms_cache_path}")

    # --- Analysis ---
    all_rms = np.array([file_rms[f] for f in files])

    print("\n=== RMS Distribution ===")
    for p in [5, 10, 20, 40, 60, 80, 90, 95, 99]:
        print(f"  P{p:2d}: {np.percentile(all_rms, p):.4f}")

    rms_low  = np.percentile(all_rms, args.train_rms_pct)
    rms_high = np.percentile(all_rms, args.test_rms_pct)
    n_train  = sum(1 for r in all_rms if r < rms_low)
    n_val    = sum(1 for r in all_rms if rms_low <= r < rms_high)
    n_test   = sum(1 for r in all_rms if r >= rms_high)

    print(f"\n=== Recommended RMS-based Split (P{args.train_rms_pct} / P{args.test_rms_pct}) ===")
    print(f"  Train (RMS < {rms_low:.4f}) : {n_train:4d} files  [{files[0]} ... {files[n_train-1]}]")
    print(f"  Val   ({rms_low:.4f} ≤ RMS < {rms_high:.4f}): {n_val:4d} files")
    print(f"  Test  (RMS ≥ {rms_high:.4f}) : {n_test:4d} files  [... {files[-1]}]")

    # --- Window label estimate ---
    baseline_rms = float(np.percentile(all_rms, 10))
    fault_thresh = baseline_rms * args.fault_factor
    n_fault_files = sum(1 for r in all_rms if r > fault_thresh)
    print(f"\n=== Window Label Estimate (fault_factor={args.fault_factor}x) ===")
    print(f"  Healthy baseline RMS (P10): {baseline_rms:.4f}")
    print(f"  Fault window threshold    : {fault_thresh:.4f}")
    print(f"  Files where file RMS > threshold: {n_fault_files} / {len(files)}")
    print(f"\n✅ Done. Run training with: python src/training/train.py")


if __name__ == "__main__":
    main()
