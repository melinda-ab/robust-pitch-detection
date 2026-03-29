import os
import re
import json
import math
import warnings
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. Configuration & Mapping
# ─────────────────────────────────────────────────────────────

NOTE_HZ = {
    'E2': 82.41,  'F2': 87.31,  'G2': 98.00,  'A2': 110.00, 'B2': 123.47,
    'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00, 
    'A3': 220.00, 'B3': 246.94, 'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 
    'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25, 
    'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99
}

@dataclass
class PipelineConfig:
    dataset_dir: str = "dataset"
    output_dir: str = "results"
    sample_rate: int = 16_000
    noise_scenarios: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "standard_mic": 10.0, 
        "high_noise": 5.0, 
    })
    wiener_alpha: float = 1.0 
    sg_n_fft: int = 2048
    sg_hop_length: int = 512
    hop_seconds: float = 0.01
    cent_tolerance: float = 50.0 

CFG = PipelineConfig()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 2. Logic Modules
# ─────────────────────────────────────────────────────────────

def apply_wiener_filter(audio, cfg):
    """Optimal Linear Filtering - Gain Masking."""
    D = librosa.stft(audio, n_fft=cfg.sg_n_fft, hop_length=cfg.sg_hop_length)
    mag_sq = np.abs(D) ** 2
    
    # Estimate noise from bottom 20% energy frames
    energy = np.mean(mag_sq, axis=0)
    Pn = np.mean(mag_sq[:, energy <= np.percentile(energy, 20)], axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.nan_to_num(np.maximum(0.0, (mag_sq - cfg.wiener_alpha * Pn) / mag_sq))
        
    return librosa.istft(gain * D, hop_length=cfg.sg_hop_length, length=len(audio)).astype(np.float32)

def detect_pitch(audio, cfg):
    hop = int(cfg.hop_seconds * cfg.sample_rate)
    f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=1000, sr=cfg.sample_rate, hop_length=hop, fill_na=0.0)
    return np.where(voiced, f0, 0.0)

def calculate_metrics(f_est, f_gt, tol):
    mask = (f_est > 0) & (f_gt > 0)
    if not np.any(mask): return 0.0, np.nan
    err = np.abs(1200.0 * np.log2(f_est[mask] / f_gt[mask]))
    rpa = np.mean(err < tol)
    return float(rpa), float(np.mean(err))

# ─────────────────────────────────────────────────────────────
# 3. Execution & Reporting
# ─────────────────────────────────────────────────────────────

def run_pipeline(cfg):
    out_path = Path(cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # This will hold all data for the JSON file
    master_report = {
        "metadata": {
            "wiener_alpha": cfg.wiener_alpha,
            "sample_rate": cfg.sample_rate,
            "cent_tolerance": cfg.cent_tolerance
        },
        "results": {}
    }

    files = list(Path(cfg.dataset_dir).glob("*.wav"))
    for p in tqdm(files, desc="Analyzing Audio"):
        match = re.search(r'([A-G]#?\d)', p.name)
        if not match: continue
        target_hz = NOTE_HZ.get(match.group(1), 0.0)
        
        # Load and Pre-process
        y, _ = librosa.load(str(p), sr=cfg.sample_rate)
        y_trim, _ = librosa.effects.trim(y, top_db=30)
        y_norm = (y_trim / np.max(np.abs(y_trim)) * 0.89).astype(np.float32)

        file_data = {}
        for label, snr in cfg.noise_scenarios.items():
            # Noise + Denoise
            noise = np.random.randn(len(y_norm)) * math.sqrt((np.mean(y_norm**2)) / (10**(snr/10)))
            noisy = np.clip(y_norm + noise, -1.0, 1.0)
            denoised = apply_wiener_filter(noisy, cfg)
            
            # Detect
            f0_b = detect_pitch(noisy, cfg)
            f0_p = detect_pitch(denoised, cfg)
            f0_gt = np.full(len(f0_b), target_hz)

            # Metrics
            rpa_b, err_b = calculate_metrics(f0_b, f0_gt, cfg.cent_tolerance)
            rpa_p, err_p = calculate_metrics(f0_p, f0_gt, cfg.cent_tolerance)

            file_data[label] = {
                "baseline": {"rpa": rpa_b, "mean_error": err_b},
                "wiener": {"rpa": rpa_p, "mean_error": err_p}
            }
            
            # (Optional) Plotting logic here if you want individual PNGs
            # ... [Previous plotting code can stay here] ...

        master_report["results"][p.stem] = file_data

    # Save the JSON file
    json_output = out_path / "final_research_report.json"
    with open(json_output, "w") as f:
        json.dump(master_report, f, indent=4)
    
    log.info(f"Pipeline finished. JSON saved to: {json_output}")

if __name__ == "__main__":
    run_pipeline(CFG)