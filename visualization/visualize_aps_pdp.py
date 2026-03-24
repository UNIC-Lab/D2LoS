"""
Visualize a single GT APS and GT PDP from .npy files.

Usage:
    python visualize_adps_gt.py --root D:/path/to/adps_root --name 630_1_117_0_128
    python visualize_adps_gt.py --root D:/path/to/adps_root --name 630_1_117_0_128 --out D:/2D
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'FreeSerif', 'STIXGeneral']
rcParams['mathtext.fontset'] = 'stix'

GRID_CONF = {
    "theta_min": -180, "theta_max": 180, "n_theta": 180,
    "tau_min_ns": 0,   "tau_max_ns": 1000,
}


def plot_pdp(npy_path, out_path):
    seq = np.load(npy_path).astype(np.float32)
    n_tau = seq.shape[0]
    tau_ns = np.linspace(GRID_CONF["tau_min_ns"], GRID_CONF["tau_max_ns"], n_tau)
    power  = seq[:, 0]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.plot(tau_ns, power, color='#f57c6e', linewidth=1.5)
    ax.set_xlabel('Delay (ns)', fontsize=14)
    ax.set_ylabel('Power (dBm)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#CCCCCC')
    ax.yaxis.grid(True, linestyle='--', alpha=0.45, color='#DDDDDD')
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"PDP saved: {out_path}")


def plot_aps(npy_path, out_path):
    seq = np.load(npy_path).astype(np.float32)
    n_theta = seq.shape[0]
    theta   = np.linspace(GRID_CONF["theta_min"], GRID_CONF["theta_max"], n_theta)
    power   = seq[:, 0]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.plot(theta, power, color='#71b8ed', linewidth=1.5)
    ax.set_xlabel('Arrival Angle (deg)', fontsize=14)
    ax.set_ylabel('Power (dBm)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#CCCCCC')
    ax.yaxis.grid(True, linestyle='--', alpha=0.45, color='#DDDDDD')
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"APS saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Root dir containing gt/pdp/ and gt/aps/ (or pdp/ and aps/) subdirs")
    parser.add_argument("--name", required=True,
                        help="Sample name, e.g. 630_1_117_0_128")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: current dir)")
    args = parser.parse_args()

    out_dir = args.out or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # Try gt/ subdir first, then directly under root
    for pdp_dir in [os.path.join(args.root, "gt", "pdp"),
                    os.path.join(args.root, "pdp")]:
        if os.path.isdir(pdp_dir):
            break

    for aps_dir in [os.path.join(args.root, "gt", "aps"),
                    os.path.join(args.root, "aps")]:
        if os.path.isdir(aps_dir):
            break

    pdp_npy = os.path.join(pdp_dir, f"pdp_{args.name}.npy")
    aps_npy = os.path.join(aps_dir, f"aps_{args.name}.npy")

    if os.path.exists(pdp_npy):
        plot_pdp(pdp_npy, os.path.join(out_dir, f"gt_pdp_{args.name}.png"))
    else:
        print(f"[skip] PDP not found: {pdp_npy}")

    if os.path.exists(aps_npy):
        plot_aps(aps_npy, os.path.join(out_dir, f"gt_aps_{args.name}.png"))
    else:
        print(f"[skip] APS not found: {aps_npy}")


if __name__ == "__main__":
    main()
