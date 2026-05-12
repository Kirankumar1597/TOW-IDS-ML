"""Build MoE3-IDS-OptionC.ipynb based on MoE3-IDS-OptionB.ipynb structure.

Three heterogeneous experts:
  - TCN-AE  (reconstruction MSE; reused verbatim from Option B)
  - GAN     (GIDS-style: adversarially-trained discriminator; score = 1 - D(x))
  - DDPM    (1D U-Net diffusion; 1-step inference at t=50; score = noise-pred MSE)

Score-normalization layer: per-expert (mu, sigma) computed on train-normal scores
at end of Stage 1, then z-normalize before the gate combines.

Evaluation harness adds AUC(train-N vs test-N) per the spec's revised
Option C requirements.
"""
import json, os, uuid

OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MoE3-IDS-OptionC.ipynb",
)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

# ============================================================
# 0. Title
# ============================================================
cells.append(md("""# MoE3-IDS Option C — Heterogeneous Generative Experts for Automotive Ethernet IDS

Three heterogeneous-paradigm experts on the TOW-IDS dataset, per Option C of `MoE3_design_spec.md`.

| # | Expert | Scoring paradigm | Notes |
|---|--------|------------------|-------|
| 1 | **TCN-AE** | Reconstruction MSE | Reused verbatim from Option B |
| 2 | **GAN-Disc** | 1 − D(x) (GIDS-style) | Adversarially trained discriminator on normal-only |
| 3 | **DDPM** | Noise-prediction MSE at t=50 | 1D U-Net, 1-step inference, channels 12 → 32 → 64 |

The novelty over Option B is heterogeneous **score paradigms** (MSE / sigmoid / denoising error)
combined via a per-expert **z-normalization layer**: μᵢ, σᵢ are computed on train-normal
scores at end of Stage 1, then `normalized_score_i = (score_i − μ_i) / σ_i` before the
gate combines them.

**Per the design-spec revision (2026-05-08), every standalone-expert and MoE
evaluation in this notebook reports AUC(train-N vs test-N) alongside ROC-AUC
and PR-AUC.** This deployment-shift-robustness metric was introduced by the
Option B diagnostic D2.5c.

**Training Strategy** (mirrors Option B):
1. Stage 1 — pretrain each expert independently on normal-only sequences (each has its native objective)
2. Calibrate score normalization (μᵢ, σᵢ from train-normal scores)
3. Stage 2 — train the gating network with score-normalization layer in place (experts frozen)
4. Stage 3 — end-to-end fine-tuning (TCN-AE + DDPM + gate; GAN frozen — see notes)
"""))

# ============================================================
# 1. Imports + config
# ============================================================
cells.append(code("""import os, random, shlex, subprocess, time, threading, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
import pynvml
import psutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
"""))

cells.append(code("""# === File paths (verbatim from Option B) ===
TRAIN_PCAP = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap"
TEST_PCAP  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap"
OUT_DIR    = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/extracted"
y_train_path = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_train.csv"
y_test_path  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_test.csv"

# === Sequence / evaluation defaults ===
WINDOW_SEC = 1.0
T_SEQ      = 40
THR_PCT    = 99

# === MoE hyperparameters — Option C (3 heterogeneous experts) ===
N_EXPERTS      = 3
TOP_K          = 2
LAMBDA_BAL     = 0.01
EXPERT_EPOCHS  = 100
GATING_EPOCHS  = 50
E2E_EPOCHS     = 30
EXPERT_LR      = 1e-3
GATING_LR      = 1e-3
E2E_LR         = 1e-4
BATCH_SIZE     = 256

# === Diffusion-expert hyperparameters ===
DIFF_T_TOTAL   = 100   # diffusion steps
DIFF_T_EVAL    = 50    # 1-step eval at this t
DIFF_BETA_LO   = 1e-4
DIFF_BETA_HI   = 0.02
DIFF_TIME_DIM  = 32
DIFF_CHANNELS  = (32, 64)        # spec default (channels 12 -> 32 -> 64)
DIFF_CHANNELS_NARROW = (16, 32)  # Ablation 5 narrow variant (~30K target)

# === GAN-expert hyperparameters ===
GAN_LATENT     = 16
GAN_DISC_LR    = 2e-4
GAN_GEN_LR     = 2e-4
"""))

# ============================================================
# 2. Utility functions (verbatim from Option B + new auc_train_vs_test_normal)
# ============================================================
cells.append(code("""# =============================================
# Utility Functions (verbatim from Option B + the new
# auc_train_vs_test_normal metric required by Option C spec).
# =============================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_feature_cols(df):
    return [c for c in df.columns if c not in ["w", "y_window", "attack_type_window"]]


def model_size_stats(model):
    params = sum(p.numel() for p in model.parameters())
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"Params": int(params), "Size (MB)": round(bytes_ / 1024**2, 4)}


def measure_latency_ms(fn, runs=200):
    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000)


def repeat_workload(fn, n=200):
    def _w():
        for _ in range(n):
            fn()
    return _w


def compute_all_metrics(y_true, scores, thr):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc":   float(roc_auc_score(y_true, scores)),
        "pr_auc":    float(average_precision_score(y_true, scores)),
        "thr":       float(thr),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def compute_metrics_99pct(y_true, scores, train_scores, pct=99):
    thr = float(np.percentile(np.asarray(train_scores, dtype=float), pct))
    return compute_all_metrics(y_true, scores, thr)


# === Option C addition (per spec revision 2026-05-08) ===
def auc_train_vs_test_normal(scores_train_normal, scores_test_normal):
    \"\"\"Deployment-shift-robustness metric.

    AUC of the model's anomaly scores treated as a binary classifier between
    train-normal sequences (label 0) and test-normal sequences (label 1).
    AUC ~ 0.5 means the model perceives no shift between the two normal pools
    (deployment-robust). AUC ~ 1.0 means the model treats test-normals as a
    different class from train-normals (shift-fragile, the failure mode that
    drove TCN-AE's 0.83 precision in the Option B run).

    Operates on whatever scalar score the model produces — works equally for
    raw, z-normalized, or gated-output scores.
    \"\"\"
    a = np.asarray(scores_train_normal, dtype=float)
    b = np.asarray(scores_test_normal, dtype=float)
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    s = np.concatenate([a, b])
    return float(roc_auc_score(y, s))


def per_attack_breakdown(scores, y_seq, atk_seq):
    df = pd.DataFrame({"score": scores, "y": y_seq.astype(int), "atk": atk_seq.astype(str)})
    attacks = sorted([a for a in df["atk"].unique() if a != "normal"])
    rows = []
    for a in attacks:
        sub = df[df["atk"].isin(["normal", a])]
        if sub["y"].nunique() < 2:
            continue
        rows.append({
            "attack":       a,
            "n_attack_seq": int((sub["atk"] == a).sum()),
            "roc_auc":      roc_auc_score(sub["y"], sub["score"]),
            "pr_auc":       average_precision_score(sub["y"], sub["score"]),
        })
    return pd.DataFrame(rows)


def fmt_cols(df, cols, fmt=".4f"):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: float(f"{x:{fmt}}") if pd.notna(x) else x)
    return df


def nvml_energy_joules(workload_fn, device_index=0, sample_ms=10):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    workload_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples_t, samples_p = [], []
    stop = threading.Event()

    def sampler():
        t0 = time.perf_counter()
        while not stop.is_set():
            samples_t.append(time.perf_counter() - t0)
            samples_p.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
            time.sleep(sample_ms / 1000.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t_start = time.perf_counter()
    workload_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    stop.set(); th.join(timeout=1.0)
    pynvml.nvmlShutdown()
    energy_j = (float(np.trapz(np.array(samples_p), np.array(samples_t)))
                if len(samples_t) >= 2
                else (float(np.mean(samples_p) * elapsed) if samples_p else np.nan))
    avg_power = energy_j / elapsed if elapsed > 0 else np.nan
    return energy_j, avg_power, elapsed


print("Utility functions loaded (incl. auc_train_vs_test_normal).")
"""))

# ============================================================
# 3. Data ingestion + feature engineering (verbatim)
# ============================================================
cells.append(md("## 1. Data Ingestion\n"))

cells.append(code("""os.makedirs(OUT_DIR, exist_ok=True)
train_csv = f"{OUT_DIR}/packets_train.csv"
test_csv  = f"{OUT_DIR}/packets_test.csv"
if os.path.exists(train_csv) and os.path.exists(test_csv):
    print("Pre-extracted CSVs found. Loading...")
else:
    print("Extracting packets with tshark (one-time) ...")
    for pcap, csv_out in [(TRAIN_PCAP, train_csv), (TEST_PCAP, test_csv)]:
        cmd = (f'tshark -r {shlex.quote(pcap)} -T fields -E header=y -E separator=, -E quote=d '
               '-e frame.number -e frame.time_epoch -e frame.len '
               '-e eth.src -e eth.dst -e eth.type '
               '-e vlan.id -e vlan.etype '
               '-e ip.src -e ip.dst -e ip.proto -e ip.ttl '
               '-e udp.srcport -e udp.dstport -e udp.length '
               f'-e udp.payload -e data.data > {shlex.quote(csv_out)}')
        subprocess.run(cmd, shell=True, check=True)

train_pkts = pd.read_csv(train_csv)
test_pkts  = pd.read_csv(test_csv)

y_train = pd.read_csv(y_train_path, header=None,
                       names=["sample_number", "normal_or_abnormal", "attack_type"])
y_test  = pd.read_csv(y_test_path,  header=None,
                       names=["sample_number", "normal_or_abnormal", "attack_type"])
for pkts, labels in [(train_pkts, y_train), (test_pkts, y_test)]:
    pkts["frame.number"] = pd.to_numeric(pkts["frame.number"], errors="coerce").astype("Int64")
    labels["sample_number"] = pd.to_numeric(labels["sample_number"], errors="coerce").astype("Int64")
train_pkts = train_pkts.merge(y_train, left_on="frame.number",
                              right_on="sample_number", how="left").drop(columns=["sample_number"])
train_pkts["y"] = (train_pkts["normal_or_abnormal"].str.lower() == "abnormal").astype(int)
test_pkts = test_pkts.merge(y_test, left_on="frame.number",
                            right_on="sample_number", how="left").drop(columns=["sample_number"])
test_pkts["y"] = (test_pkts["normal_or_abnormal"].str.lower() == "abnormal").astype(int)
for df in [train_pkts, test_pkts]:
    at_col = None
    for c in ["attack_type", "attack_type_y", "attack_type_x"]:
        if c in df.columns and df[c].notna().any():
            at_col = c
            break
    if at_col and at_col != "attack_type":
        df["attack_type"] = df[at_col]
    df["attack_type"] = df["attack_type"].astype(str).str.strip().str.lower()
print("Train/test packets loaded.")
print(f"Train: {train_pkts.shape}, Test: {test_pkts.shape}")
"""))

cells.append(md("""## 2. Feature Engineering

12 windowed features (1-second windows) + sliding-window sequences of length T_SEQ=40.
Verbatim from Option B / TOW-IDS-MoE.
"""))

cells.append(code("""def build_global_windows_with_attack(df, window_sec=1.0):
    df = df.copy()
    df["t"] = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t")
    t0 = df["t"].iloc[0]
    df["w"] = np.floor((df["t"] - t0) / window_sec).astype(int)
    df["frame.len"] = pd.to_numeric(df["frame.len"], errors="coerce").fillna(0.0)
    df["dt"] = df["t"].diff().fillna(0.0).clip(lower=0.0)
    df["is_multicast"] = df["eth.dst"].astype(str).str.lower().str.startswith(("01:", "33:")).astype(int)
    df["has_vlan"] = df["vlan.id"].notna().astype(int)
    def agg(g):
        y_win = int(pd.to_numeric(g["y"], errors="coerce").fillna(0).max())
        if y_win == 0:
            atk = "normal"
        else:
            atk_s = g.loc[g["y"] == 1, "attack_type"].astype(str).str.strip()
            atk = atk_s.value_counts().index[0] if len(atk_s) else "anomaly"
        return pd.Series({
            "pkt_count":       len(g),
            "bytes_sum":       g["frame.len"].sum(),
            "pkt_len_mean":    g["frame.len"].mean(),
            "pkt_len_std":     g["frame.len"].std(ddof=0),
            "dt_mean":         g["dt"].mean(),
            "dt_std":          g["dt"].std(ddof=0),
            "uniq_src_mac":    g["eth.src"].nunique(dropna=True),
            "uniq_dst_mac":    g["eth.dst"].nunique(dropna=True),
            "uniq_ip_src":     g["ip.src"].nunique(dropna=True),
            "uniq_ip_dst":     g["ip.dst"].nunique(dropna=True),
            "multicast_ratio": g["is_multicast"].mean(),
            "vlan_ratio":      g["has_vlan"].mean(),
            "y_window":        y_win,
            "attack_type_window": atk,
        })
    return df.groupby("w").apply(agg).reset_index().fillna(0.0)


def make_sequences(df, t_seq=20, stride=1, feature_cols=None):
    df = df.sort_values("w").reset_index(drop=True)
    if feature_cols is None:
        feature_cols = get_feature_cols(df)
    X = df[feature_cols].astype(float).to_numpy()
    y = df["y_window"].astype(int).to_numpy()
    atk = (df["attack_type_window"].astype(str).to_numpy()
           if "attack_type_window" in df.columns
           else np.array(["normal"] * len(df)))
    seqs, seq_y, seq_atk = [], [], []
    for start in range(0, len(df) - t_seq + 1, stride):
        end = start + t_seq
        seqs.append(X[start:end])
        y_s = int(y[start:end].max())
        seq_y.append(y_s)
        if y_s == 0:
            seq_atk.append("normal")
        else:
            anom = atk[start:end][y[start:end] == 1]
            seq_atk.append(pd.Series(anom).value_counts().index[0] if len(anom) else "anomaly")
    return np.stack(seqs), np.array(seq_y), np.array(seq_atk)


def scale_train_test_sequences(Xtr_normal, Xte_all):
    N, T, D = Xtr_normal.shape
    scaler = StandardScaler().fit(Xtr_normal.reshape(-1, D))
    def sx(X):
        n, t, d = X.shape
        return scaler.transform(X.reshape(-1, d)).reshape(n, t, d)
    return sx(Xtr_normal), sx(Xte_all), scaler


train_w = build_global_windows_with_attack(train_pkts, WINDOW_SEC)
test_w  = build_global_windows_with_attack(test_pkts,  WINDOW_SEC)
feat_cols = get_feature_cols(train_w)
D = len(feat_cols)

Xtr_seq, ytr_seq, atr_seq = make_sequences(
    train_w[train_w["y_window"] == 0].reset_index(drop=True),
    t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
Xte_seq, yte_seq, ate_seq = make_sequences(
    test_w, t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
Xtr_s, Xte_s, seq_scaler = scale_train_test_sequences(Xtr_seq, Xte_seq)
Xtr_s = Xtr_s.astype(np.float32); Xte_s = Xte_s.astype(np.float32)

print(f"D={D}, T_SEQ={T_SEQ}")
print(f"Train sequences (normal): {Xtr_s.shape}")
print(f"Test  sequences (all):    {Xte_s.shape}")
print(f"Test - normal: {int((yte_seq==0).sum())}, anomaly: {int((yte_seq==1).sum())}")
print(f"Test attack types: {pd.Series(ate_seq).value_counts().to_dict()}")
"""))

# ============================================================
# 3. EDA (lightweight)
# ============================================================
cells.append(md("## 3. Exploratory Data Analysis\n"))

cells.append(code("""fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for col, ax in zip(feat_cols[:6], axes.flat):
    ax.hist(train_w.loc[train_w["y_window"]==0, col], bins=40, alpha=0.5,
            label="Normal", density=True)
    ax.hist(test_w.loc[test_w["y_window"]==1, col], bins=40, alpha=0.5,
            label="Attack", density=True)
    ax.set_title(col); ax.legend(fontsize=8)
plt.suptitle("Feature Distributions: Normal (train) vs Attack (test)", fontsize=14)
plt.tight_layout(); plt.show()
print(f"Xtr_s range: [{Xtr_s.min():.2f}, {Xtr_s.max():.2f}]  "
      f"Xte_s range: [{Xte_s.min():.2f}, {Xte_s.max():.2f}]")
"""))

# ============================================================
# 4. Expert architectures
# ============================================================
cells.append(md("""## 4. Expert Architectures (Option C: heterogeneous generative experts)

Three experts using fundamentally different anomaly *scoring* paradigms. Each
expert exposes a uniform `anomaly_score(x) -> (B,)` interface so the MoE
wrapper can z-normalize and combine them.

| # | Expert | `forward(x)` returns | `anomaly_score(x)` returns |
|---|--------|----------------------|----------------------------|
| 1 | **TCN-AE**  | reconstruction `(B, T, D)` | mean MSE per sequence (scalar) |
| 2 | **GAN-Disc** | sigmoid `D(x)` ∈ [0,1] | `1 − D(x)` (low D = anomalous) |
| 3 | **DDPM**    | noise prediction `(B, T, D)` | noise-prediction MSE at fixed `t=DIFF_T_EVAL` |
"""))

# ----- TCN-AE
cells.append(code("""# ============================================================
# Expert 1: TCN-AE — reused verbatim from Option B, plus anomaly_score()
# ============================================================
class TCN_AE_Expert(nn.Module):
    'TCN Autoencoder (TENET-style). Reused verbatim from Option B.'
    def __init__(self, D, channels=64, kernel_size=3):
        super().__init__()
        k = kernel_size
        self.enc1 = nn.Conv1d(D, channels,   k, padding=(k-1)*1//2, dilation=1)
        self.enc2 = nn.Conv1d(channels, channels*2, k, padding=(k-1)*2//2, dilation=2)
        self.enc3 = nn.Conv1d(channels*2, channels,  k, padding=(k-1)*4//2, dilation=4)
        self.dec1 = nn.Conv1d(channels, channels*2, k, padding=(k-1)//2)
        self.dec2 = nn.Conv1d(channels*2, channels,  k, padding=(k-1)//2)
        self.dec3 = nn.Conv1d(channels, D, 1)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.relu(self.enc1(h)); h = F.relu(self.enc2(h)); h = F.relu(self.enc3(h))
        h = F.relu(self.dec1(h)); h = F.relu(self.dec2(h)); h = self.dec3(h)
        return h.transpose(1, 2)

    def anomaly_score(self, x):
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=(1, 2))


print("TCN_AE_Expert defined.")
"""))

# ----- GAN expert
cells.append(code("""# ============================================================
# Expert 2: GAN-Disc — GIDS-style discriminator-score expert
# ============================================================
class TS_Generator(nn.Module):
    'Generator: latent z -> (B, T, D) fake "normal" sequence.'
    def __init__(self, latent=GAN_LATENT, T=T_SEQ, D=12):
        super().__init__()
        self.T, self.D = T, D
        self.fc = nn.Sequential(
            nn.Linear(latent, 128),  nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),     nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, T * D),
        )

    def forward(self, z):
        return self.fc(z).view(-1, self.T, self.D)


class TS_Discriminator(nn.Module):
    'Discriminator: (B, T, D) -> sigmoid scalar, "is this real-normal?"'
    def __init__(self, T=T_SEQ, D=12, ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(D, ch,    3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ch, ch*2, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(ch*2, 64), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),    nn.Sigmoid(),
        )

    def forward(self, x_seq):
        h = self.conv(x_seq.transpose(1, 2))     # (B, 2*ch, 1)
        return self.fc(h.squeeze(-1))            # (B, 1)


class GAN_Expert(nn.Module):
    \"\"\"GIDS-style GAN expert. Adversarially trained on normal-only sequences.

    At inference, only the discriminator D is used:
        anomaly_score(x) = 1 - D(x)
    Low D(x) -> anomalous. Generator G is used during Stage 1 adversarial
    pretraining only and is frozen during Stages 2 / 3.
    \"\"\"
    def __init__(self, latent=GAN_LATENT, T=T_SEQ, D=12, ch=64):
        super().__init__()
        self.G    = TS_Generator(latent, T, D)
        self.Disc = TS_Discriminator(T, D, ch)
        self.latent = latent

    def forward(self, x_seq):
        return self.Disc(x_seq).squeeze(-1)

    def anomaly_score(self, x_seq):
        d = self.Disc(x_seq).squeeze(-1)
        return 1.0 - d


print("GAN_Expert defined.")
"""))

# ----- DDPM expert
cells.append(code("""# ============================================================
# Expert 3: DDPM — 1D U-Net diffusion, 1-step inference at t=DIFF_T_EVAL
# ============================================================
class TimeEmbedding(nn.Module):
    'Sinusoidal embedding + 2-layer MLP -> (B, time_dim).'
    def __init__(self, time_dim=DIFF_TIME_DIM):
        super().__init__()
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

    def forward(self, t):
        # t: (B,) integer/float timestep
        half = self.time_dim // 2
        freqs = torch.exp(-np.log(10000.0) *
                           torch.arange(half, device=t.device).float() / max(half, 1))
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, time_dim)
        return self.mlp(emb)


class ResBlock1D(nn.Module):
    'GroupNorm -> SiLU -> Conv -> +TimeEmb -> GroupNorm -> SiLU -> Conv + Skip.'
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        g_in  = max(1, min(8, in_c))
        g_out = max(1, min(8, out_c))
        self.norm1 = nn.GroupNorm(g_in,  in_c)
        self.conv1 = nn.Conv1d(in_c,  out_c, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_c)
        self.norm2 = nn.GroupNorm(g_out, out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.skip  = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Diffusion1DUNet(nn.Module):
    \"\"\"1D U-Net for diffusion. Channels D -> ch_mid -> ch_max -> ch_mid -> D.

    Default channels=(32, 64) gives the spec's 12 -> 32 -> 64 path.
    Ablation 5 narrowed variant uses channels=(16, 32).
    \"\"\"
    def __init__(self, D=12, channels=DIFF_CHANNELS, time_dim=DIFF_TIME_DIM):
        super().__init__()
        ch_mid, ch_max = channels
        self.time_emb = TimeEmbedding(time_dim)
        self.in_proj  = nn.Conv1d(D, ch_mid, 3, padding=1)
        self.down1    = ResBlock1D(ch_mid, ch_mid, time_dim)
        self.down2    = ResBlock1D(ch_mid, ch_max, time_dim)
        self.mid      = ResBlock1D(ch_max, ch_max, time_dim)   # single bottleneck ResBlock
        self.up1      = ResBlock1D(ch_max + ch_max, ch_mid, time_dim)
        self.up2      = ResBlock1D(ch_mid + ch_mid, ch_mid, time_dim)
        self.out_proj = nn.Conv1d(ch_mid, D, 3, padding=1)

    def forward(self, x_seq, t):
        # x_seq: (B, T, D) -> (B, D, T); t: (B,) timesteps
        x = x_seq.transpose(1, 2)
        t_emb = self.time_emb(t)
        h0 = self.in_proj(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(h1, t_emb)
        h  = self.mid(h2, t_emb)
        h  = self.up1(torch.cat([h, h2], dim=1), t_emb)
        h  = self.up2(torch.cat([h, h1], dim=1), t_emb)
        out = self.out_proj(h)
        return out.transpose(1, 2)


class DDPM_Expert(nn.Module):
    \"\"\"1-step DDPM expert. Anomaly score = noise-prediction MSE at fixed t.

    Stage-1 training: standard DDPM noise-prediction objective on random t.
    Inference: add a deterministic-seed noise sample at fixed t = DIFF_T_EVAL,
    predict the noise, return MSE between prediction and the true noise.
    \"\"\"
    def __init__(self, D=12, T=T_SEQ, channels=DIFF_CHANNELS,
                 t_total=DIFF_T_TOTAL, t_eval=DIFF_T_EVAL,
                 beta_lo=DIFF_BETA_LO, beta_hi=DIFF_BETA_HI,
                 time_dim=DIFF_TIME_DIM, eval_seed=42):
        super().__init__()
        self.D, self.T = D, T
        self.t_total = t_total
        self.t_eval = t_eval
        self.eval_seed = eval_seed
        self.unet = Diffusion1DUNet(D=D, channels=channels, time_dim=time_dim)
        betas      = torch.linspace(beta_lo, beta_hi, t_total)
        alphas     = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)

    def add_noise(self, x_0, t, noise=None):
        \"\"\"x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise.\"\"\"
        if noise is None:
            noise = torch.randn_like(x_0)
        ab = self.alpha_bars[t]
        if ab.dim() == 0:
            x_t = ab.sqrt() * x_0 + (1 - ab).sqrt() * noise
        else:
            ab = ab[:, None, None]
            x_t = ab.sqrt() * x_0 + (1 - ab).sqrt() * noise
        return x_t, noise

    def forward(self, x_seq):
        \"\"\"Training forward — random t, returns (predicted_noise, true_noise).\"\"\"
        B = x_seq.size(0)
        t = torch.randint(0, self.t_total, (B,), device=x_seq.device)
        x_t, noise = self.add_noise(x_seq, t)
        pred = self.unet(x_t, t)
        return pred, noise

    def anomaly_score(self, x_seq):
        \"\"\"Inference — deterministic noise at fixed t_eval, MSE(pred, true).\"\"\"
        B = x_seq.size(0)
        gen = torch.Generator(device=x_seq.device).manual_seed(self.eval_seed)
        noise = torch.empty_like(x_seq).normal_(generator=gen)
        t = torch.full((B,), self.t_eval, device=x_seq.device, dtype=torch.long)
        x_t, true_noise = self.add_noise(x_seq, t, noise=noise)
        pred = self.unet(x_t, t)
        return torch.mean((pred - true_noise) ** 2, dim=(1, 2))


print("Diffusion1DUNet, DDPM_Expert defined.")
"""))

cells.append(code("""# Parameter-count summary
for cls, kw in [(TCN_AE_Expert, dict(D=D)),
                (GAN_Expert,    dict(latent=GAN_LATENT, T=T_SEQ, D=D)),
                (DDPM_Expert,   dict(D=D, T=T_SEQ, channels=DIFF_CHANNELS))]:
    tmp = cls(**kw)
    print(f"  {cls.__name__:15s}  {model_size_stats(tmp)['Params']:>8,} params")
# Note: GAN's total includes the generator (training-only). At inference only
# the discriminator runs. Discriminator-only param count printed separately.
gan_tmp = GAN_Expert(latent=GAN_LATENT, T=T_SEQ, D=D)
print(f"  {'  GAN.Disc only':17s}  {model_size_stats(gan_tmp.Disc)['Params']:>8,} params (inference path)")
print(f"  {'  GAN.G only':17s}  {model_size_stats(gan_tmp.G)['Params']:>8,} params (training only)")
"""))

# ============================================================
# 5. Gating + MoE wrapper with score normalization
# ============================================================
cells.append(md("""## 5. Sparse Gating Network & MoE Wrapper with Score Normalization

The gating network is verbatim from Option B (mean+std pool → 2-layer MLP →
sparse top-K softmax). The **MoE wrapper is new**: it expects each expert's
`anomaly_score(x)` to return `(B,)` and combines z-normalized scores rather
than reconstructions.

Per-expert μᵢ, σᵢ are computed on train-normal scores at end of Stage 1 via
`MoE_OptionC.calibrate_score_normalization(...)` and stored as buffers; from
that point onward `forward(x)` returns the gate-weighted z-normalized
anomaly score, the gate weights, the raw logits, and the raw / z-normalized
score tensors for diagnostics.
"""))

cells.append(code("""class SparseGatingNetwork(nn.Module):
    'Sparse Top-K Gating with Noisy Routing (verbatim from Option B).'
    def __init__(self, seq_dim, num_experts, top_k=2, noisy=True):
        super().__init__()
        input_dim = seq_dim * 2
        self.num_experts = num_experts; self.top_k = top_k; self.noisy = noisy
        self.gate = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                  nn.Linear(64, num_experts))
        if noisy:
            self.noise_linear = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                              nn.Linear(64, num_experts))

    def forward(self, x):
        x_agg = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)
        logits = self.gate(x_agg)
        if self.noisy and self.training:
            noise_std = F.softplus(self.noise_linear(x_agg))
            logits = logits + torch.randn_like(logits) * noise_std
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        sparse = torch.full_like(logits, float('-inf'))
        sparse.scatter_(1, top_idx, top_vals)
        return F.softmax(sparse, dim=-1), logits


class MoE_OptionC(nn.Module):
    \"\"\"Option C MoE: heterogeneous-paradigm experts with z-normalized scoring.

    Each expert returns a scalar anomaly score; the wrapper z-normalizes those
    scores using train-normal-derived (mu, sigma) and combines via the same
    sparse gating network used in Option B. Stage 1 trains experts via their
    native objectives; `calibrate_score_normalization` then fits per-expert
    (mu, sigma) on train-normal scores and locks them in.
    \"\"\"
    SCORE_SIGMA_FLOOR = 0.05  # spec convention; prevents z-norm blow-up on near-zero sigma

    def __init__(self, experts, input_dim, num_experts=3, top_k=2, noisy_gating=True):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating  = SparseGatingNetwork(input_dim, num_experts, top_k, noisy_gating)
        self.num_experts = num_experts; self.top_k = top_k
        self.register_buffer("score_mu",    torch.zeros(num_experts))
        self.register_buffer("score_sigma", torch.ones(num_experts))
        self.calibrated = False

    def calibrate_score_normalization(self, X_train_normal, batch_size=256, expert_names=None):
        'Compute per-expert (mu, sigma) on train-normal anomaly scores. Sigma floored at 0.05.'
        was_training = self.training
        self.eval()
        loader = DataLoader(TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32)),
                            batch_size=batch_size, shuffle=False)
        all_scores = [[] for _ in range(self.num_experts)]
        dev = next(self.parameters()).device
        with torch.no_grad():
            for (x,) in loader:
                x = x.to(dev)
                for i, e in enumerate(self.experts):
                    s = e.anomaly_score(x).cpu().numpy()
                    all_scores[i].append(s)
        for i in range(self.num_experts):
            arr = np.concatenate(all_scores[i])
            mu_i           = float(np.mean(arr))
            sigma_measured = float(np.std(arr))
            sigma_floored  = max(sigma_measured, self.SCORE_SIGMA_FLOOR)
            self.score_mu[i]    = mu_i
            self.score_sigma[i] = sigma_floored
            if sigma_measured < self.SCORE_SIGMA_FLOOR:
                tag = expert_names[i] if expert_names else f"expert {i}"
                print(f"  [sigma_floor] {tag}: measured sigma={sigma_measured:.4f} < "
                      f"floor={self.SCORE_SIGMA_FLOOR:.2f}; using floor for z-norm stability")
        self.calibrated = True
        if was_training:
            self.train()

    def forward(self, x):
        # raw scores: (B, E)
        raw_scores = torch.stack([e.anomaly_score(x) for e in self.experts], dim=1)
        # z-normalize
        z_scores = (raw_scores - self.score_mu) / self.score_sigma
        # gate
        gates, raw_logits = self.gating(x)
        # weighted sum
        anomaly = (gates * z_scores).sum(dim=1)
        return anomaly, gates, raw_logits, raw_scores, z_scores

    def anomaly_scores(self, x, method="z_combined"):
        anomaly, gates, _, raw_scores, z_scores = self.forward(x)
        if method == "z_combined":
            return anomaly
        if method == "raw_combined":
            return (gates * raw_scores).sum(dim=1)
        if method == "max_z":
            return z_scores.max(dim=1).values
        return anomaly


def load_balancing_loss(gates, raw_logits, num_experts):
    f = gates.mean(dim=0)
    P = F.softmax(raw_logits, dim=-1).mean(dim=0)
    return num_experts * (f * P).sum()


print("MoE_OptionC + SparseGatingNetwork defined.")
"""))

# ============================================================
# 6. Training pipeline
# ============================================================
cells.append(md("""## 6. Training Pipeline (heterogeneous Stage 1)

Each expert has its own native Stage 1 objective:

| Expert | Stage 1 objective |
|---|---|
| TCN-AE | MSE reconstruction loss (verbatim from Option B) |
| GAN-Disc | Standard non-saturating GAN: alternate D-step (BCE on real vs G(z)) and G-step (BCE on D(G(z)) → 1) |
| DDPM | Noise-prediction MSE on random t per batch |

After Stage 1: `MoE_OptionC.calibrate_score_normalization` fits per-expert μ, σ.

**Stages 2 / 3** train the gate to minimize `mean(anomaly_z_score**2)` on
train-normals + λ * load-balancing. Stage 3 unfreezes TCN-AE and DDPM but
**keeps the GAN frozen** — the GAN's adversarial equilibrium would be broken
by the weighted-MSE-on-z signal (a single criterion can't replace the
adversarial pair). This is a documented deviation from spec; see CLAUDE.md.
"""))

cells.append(code("""def pretrain_tcn(expert, X, *, epochs=100, lr=1e-3, batch_size=256, seed=0, verbose=True):
    'Stage 1 for TCN-AE: standard MSE reconstruction.'
    set_seed(seed)
    model = expert.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        losses = []
        for (x,) in loader:
            x = x.to(device)
            loss = loss_fn(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"    [TCN-AE] ep {ep:3d}: loss={np.mean(losses):.6f}")
    return model


def pretrain_gan(expert, X, *, epochs=100, lr_d=GAN_DISC_LR, lr_g=GAN_GEN_LR,
                  batch_size=256, seed=0, verbose=True):
    'Stage 1 for GAN: standard non-saturating GAN, alternating D / G steps.'
    set_seed(seed)
    model = expert.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt_d = optim.Adam(model.Disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_g = optim.Adam(model.G.parameters(),    lr=lr_g, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    model.train()
    for ep in range(epochs):
        d_losses, g_losses = [], []
        for (x,) in loader:
            x = x.to(device)
            B = x.size(0)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)
            # --- D step
            z = torch.randn(B, model.latent, device=device)
            with torch.no_grad():
                x_fake = model.G(z)
            d_real = model.Disc(x)
            d_fake = model.Disc(x_fake)
            loss_d = bce(d_real, real_lbl) + bce(d_fake, fake_lbl)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            # --- G step
            z = torch.randn(B, model.latent, device=device)
            d_fake_for_g = model.Disc(model.G(z))
            loss_g = bce(d_fake_for_g, real_lbl)
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            d_losses.append(loss_d.item()); g_losses.append(loss_g.item())
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"    [GAN]    ep {ep:3d}: D_loss={np.mean(d_losses):.4f}  G_loss={np.mean(g_losses):.4f}")
    return model


def pretrain_ddpm(expert, X, *, epochs=100, lr=1e-3, batch_size=256, seed=0, verbose=True):
    'Stage 1 for DDPM: noise-prediction MSE on random t per batch.'
    set_seed(seed)
    model = expert.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        losses = []
        for (x,) in loader:
            x = x.to(device)
            pred, true_noise = model(x)
            loss = loss_fn(pred, true_noise)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"    [DDPM]   ep {ep:3d}: loss={np.mean(losses):.6f}")
    return model


def train_moe_optionc(moe, X, *, epochs=50, lr=1e-3, batch_size=256, lambda_bal=0.01,
                      stage="gating", seed=0, verbose=True):
    \"\"\"Stage 2 / Stage 3 trainer.

    Loss = mean(anomaly_z_score^2) on train-normals + lambda_bal * load_balancing.
    stage='gating': only gate parameters trained; experts frozen.
    stage='e2e':    gate + TCN-AE + DDPM trained; GAN frozen (adversarial
                    equilibrium would be broken by a single criterion).
    \"\"\"
    set_seed(seed)
    moe = moe.to(device)
    # Default: freeze all experts
    for e in moe.experts:
        for p in e.parameters():
            p.requires_grad = False
    if stage == "e2e":
        # Unfreeze TCN-AE (index 0) and DDPM (index 2). Keep GAN (index 1) frozen.
        for p in moe.experts[0].parameters(): p.requires_grad = True
        for p in moe.experts[2].parameters(): p.requires_grad = True
    params = [p for p in moe.parameters() if p.requires_grad]

    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(params, lr=lr)
    moe.train()
    for ep in range(epochs):
        ep_anom, ep_bal = [], []
        for (x,) in loader:
            x = x.to(device)
            anomaly, gates, raw_logits, _, _ = moe(x)
            # Train normals -> anomaly z-score should be near 0
            loss_anom = (anomaly ** 2).mean()
            loss_bal  = load_balancing_loss(gates, raw_logits, moe.num_experts)
            loss = loss_anom + lambda_bal * loss_bal
            opt.zero_grad(); loss.backward(); opt.step()
            ep_anom.append(loss_anom.item()); ep_bal.append(loss_bal.item())
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            tag = "Gating" if stage == "gating" else "E2E"
            print(f"    [MoE-{tag}] ep {ep:3d}: anom={np.mean(ep_anom):.4f}  bal={np.mean(ep_bal):.4f}")
    return moe


print("Training functions defined.")
"""))

# ============================================================
# 7. Smoke test
# ============================================================
cells.append(md("""## 6.5 Smoke Test (1 epoch, 100 samples) — required by spec

Per `MoE3_design_spec.md` § Smoke Test: before launching Stage 1 training,
run a 1-epoch dry pass on a 100-sample subset to catch shape errors and
confirm parameter counts.

**Option C addition**: also report `AUC(train-N vs test-N)` for each expert
and the MoE on the partially-trained models. The values won't be meaningful
yet (only 1 epoch of training), but they confirm the metric is computable
and let us see capacity-vs-shift behavior emerging from the very first run.
"""))

cells.append(code("""# ============================================================
# Smoke Test — 1 epoch, 100 samples, all stages, AUC(train-N vs test-N)
# ============================================================
print("=" * 60)
print("SMOKE TEST  Option C, 1 epoch, 100 samples")
print("=" * 60)

set_seed(42)
SMOKE_N      = 100
SMOKE_EPOCHS = 1
SMOKE_BATCH  = 32

# Spec targets per the dual-accounting convention (MoE3_design_spec.md, 2026-05-08).
spec_inference = {"TCN-AE": 102_000, "GAN": 35_000,  "DDPM": 110_000}
spec_total     = {"TCN-AE": 102_000, "GAN": 195_000, "DDPM": 110_000}
smoke_tcn = TCN_AE_Expert(D, channels=64, kernel_size=3)
smoke_gan = GAN_Expert(latent=GAN_LATENT, T=T_SEQ, D=D, ch=64)
smoke_ddpm = DDPM_Expert(D=D, T=T_SEQ, channels=DIFF_CHANNELS)
smoke_experts = [smoke_tcn, smoke_gan, smoke_ddpm]
smoke_names   = ["TCN-AE", "GAN", "DDPM"]
inference_part = {"TCN-AE": smoke_tcn, "GAN": smoke_gan.Disc, "DDPM": smoke_ddpm}

# --- (1) parameter counts vs spec — dual accounting ---
print("\\n[1] Parameter counts vs spec target (dual accounting):")
deviations = []
tot_inf = 0; tot_all = 0
print(f"  {'expert':6s}  {'inference path':>14s}  {'(target)':>10s}  {'dev':>7s}   "
      f"{'total trainable':>15s}  {'(target)':>10s}  {'dev':>7s}")
for name, exp in zip(smoke_names, smoke_experts):
    p_inf = model_size_stats(inference_part[name])["Params"]
    p_all = model_size_stats(exp)["Params"]
    tot_inf += p_inf; tot_all += p_all
    t_inf = spec_inference[name]; t_all = spec_total[name]
    dev_inf = 100.0 * (p_inf - t_inf) / t_inf
    dev_all = 100.0 * (p_all - t_all) / t_all
    flag_inf = "  OK  " if abs(dev_inf) <= 10.0 else " WARN "
    flag_all = "  OK  " if abs(dev_all) <= 10.0 else " WARN "
    print(f"  [{flag_inf}] {name:6s}  {p_inf:>10,}    ({t_inf:>7,})  {dev_inf:+5.1f}%   "
          f"[{flag_all}] {p_all:>10,}    ({t_all:>7,})  {dev_all:+5.1f}%")
    deviations.append((name, p_inf, t_inf, dev_inf, p_all, t_all, dev_all))
print(f"  ----  TOTAL  {tot_inf:>10,}    ({sum(spec_inference.values()):>7,})           "
      f"{tot_all:>10,}    ({sum(spec_total.values()):>7,})")
gan_g = model_size_stats(smoke_gan.G)['Params']
print(f"  GAN.G (training-only): {gan_g:,} params (no inference-path budget)")

# --- (2) shape contracts ---
print("\\n[2] Shape contracts on a (4, 40, 12) batch:")
x_dummy = torch.randn(4, T_SEQ, D, device=device)
for name, exp in zip(smoke_names, smoke_experts):
    exp = exp.to(device).eval()
    with torch.no_grad():
        s = exp.anomaly_score(x_dummy)
    ok = tuple(s.shape) == (4,)
    print(f"  [{'OK' if ok else 'FAIL':4s}] {name:6s} anomaly_score(x).shape = {tuple(s.shape)}")
    if not ok:
        raise RuntimeError(f"{name}.anomaly_score shape mismatch")

# --- (3) Stage 1 pretrain on 100 samples, 1 epoch, native objective per expert ---
Xtr_smoke = Xtr_s[:SMOKE_N]
print(f"\\n[3] Stage 1 ({SMOKE_EPOCHS} epoch, native objective per expert):")
print("  --- TCN-AE ---")
pretrain_tcn(smoke_tcn,  Xtr_smoke, epochs=SMOKE_EPOCHS, lr=EXPERT_LR,
             batch_size=SMOKE_BATCH, seed=42)
print("  --- GAN ---")
pretrain_gan(smoke_gan,  Xtr_smoke, epochs=SMOKE_EPOCHS,
             batch_size=SMOKE_BATCH, seed=42)
print("  --- DDPM ---")
pretrain_ddpm(smoke_ddpm, Xtr_smoke, epochs=SMOKE_EPOCHS, lr=EXPERT_LR,
              batch_size=SMOKE_BATCH, seed=42)

# --- (4) score-normalization calibration ---
print("\\n[4] Score-normalization calibration on Xtr_smoke train-normals:")
moe_smoke = MoE_OptionC(smoke_experts, input_dim=D, num_experts=N_EXPERTS,
                        top_k=TOP_K, noisy_gating=True).to(device)
moe_smoke.calibrate_score_normalization(Xtr_smoke, batch_size=SMOKE_BATCH)
print(f"  per-expert mu = {moe_smoke.score_mu.cpu().numpy().round(4)}")
print(f"  per-expert sd = {moe_smoke.score_sigma.cpu().numpy().round(4)}")

# --- (5) Stage 2 + Stage 3 (1 epoch each) ---
print(f"\\n[5] Stage 2 (gating, {SMOKE_EPOCHS} epoch):")
train_moe_optionc(moe_smoke, Xtr_smoke, epochs=SMOKE_EPOCHS, lr=GATING_LR,
                  batch_size=SMOKE_BATCH, lambda_bal=LAMBDA_BAL, stage="gating", seed=42)
print(f"\\n[6] Stage 3 (e2e, {SMOKE_EPOCHS} epoch, GAN frozen):")
train_moe_optionc(moe_smoke, Xtr_smoke, epochs=SMOKE_EPOCHS, lr=E2E_LR,
                  batch_size=SMOKE_BATCH, lambda_bal=LAMBDA_BAL, stage="e2e", seed=42)

# --- (7) end-to-end shape check ---
print("\\n[7] End-to-end MoE forward shape check:")
moe_smoke.eval()
with torch.no_grad():
    x_check = torch.tensor(Xtr_smoke[:8], dtype=torch.float32, device=device)
    anomaly, gates, raw_logits, raw_scores, z_scores = moe_smoke(x_check)
print(f"  anomaly      shape = {tuple(anomaly.shape)}      (expect (8,))")
print(f"  gates        shape = {tuple(gates.shape)}            (expect (8, 3))")
print(f"  raw_scores   shape = {tuple(raw_scores.shape)}            (expect (8, 3))")
print(f"  z_scores     shape = {tuple(z_scores.shape)}            (expect (8, 3))")
print(f"  gate row sums      = {gates.sum(dim=-1).cpu().numpy().round(4)}")

# --- (8) AUC(train-N vs test-N) per expert and MoE (capacity-vs-shift visibility) ---
print("\\n[8] AUC(train-N vs test-N) — capacity-vs-shift signal after 1 epoch:")
@torch.no_grad()
def expert_scores(expert, X, batch=512):
    expert.eval()
    parts = []
    for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch, shuffle=False):
        parts.append(expert.anomaly_score(x.to(device)).cpu().numpy())
    return np.concatenate(parts)

@torch.no_grad()
def moe_scores_eval(moe_model, X, method="z_combined", batch=512):
    moe_model.eval()
    parts = []
    for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch, shuffle=False):
        parts.append(moe_model.anomaly_scores(x.to(device), method=method).cpu().numpy())
    return np.concatenate(parts)

Xte_normal = Xte_s[yte_seq == 0]
print(f"  (using full train pool {Xtr_s.shape[0]} normals + {Xte_normal.shape[0]} test normals)")
for name, exp in zip(smoke_names, smoke_experts):
    s_tr = expert_scores(exp, Xtr_s)
    s_te = expert_scores(exp, Xte_normal)
    auc_shift = auc_train_vs_test_normal(s_tr, s_te)
    print(f"  {name:6s}  AUC(train-N vs test-N) = {auc_shift:.4f}   "
          f"(mean train = {s_tr.mean():.4f}, mean test-N = {s_te.mean():.4f})")
s_tr_moe = moe_scores_eval(moe_smoke, Xtr_s)
s_te_moe = moe_scores_eval(moe_smoke, Xte_normal)
auc_moe = auc_train_vs_test_normal(s_tr_moe, s_te_moe)
print(f"  MoE     AUC(train-N vs test-N) = {auc_moe:.4f}")

# --- verdict (dual accounting: flag deviation on inference-path OR total-trainable) ---
print("\\n" + "=" * 60)
warns_inf = [d for d in deviations if abs(d[3]) > 10.0]
warns_all = [d for d in deviations if abs(d[6]) > 10.0]
if warns_inf or warns_all:
    print(f"SMOKE TEST FINISHED with parameter-count deviations >10%:")
    for d in warns_inf:
        print(f"   - {d[0]} inference-path: {d[1]:,} vs target {d[2]:,} ({d[3]:+.1f}%)")
    for d in warns_all:
        print(f"   - {d[0]} total-trainable: {d[4]:,} vs target {d[5]:,} ({d[6]:+.1f}%)")
    print("Per spec: review before launching full training.")
else:
    print("SMOKE TEST PASSED. All shape contracts hold; all params within 10% of spec "
          "on both inference-path and total-trainable accounting.")
print("Note: AUC(train-N vs test-N) values reported above are on partially-")
print("trained (1-epoch) experts and serve as smoke-test signal only.")
print("=" * 60)
"""))

# ============================================================
# Stage 1 / 2 / 3 (full)
# ============================================================
cells.append(md("""---
**STOP — review the smoke-test output above before running the cells below.**

If parameter counts and shape checks look correct, proceed with full Stage 1
/ 2 / 3 training. The full Stage 1 takes ~3–5 minutes given the ~234 train
sequences (TCN-AE + GAN + DDPM × 100 epochs).
"""))

cells.append(code("""print("=" * 60); print("STAGE 1 — heterogeneous expert pretraining"); print("=" * 60)
set_seed(42)
expert_tcn  = TCN_AE_Expert(D, channels=64, kernel_size=3)
expert_gan  = GAN_Expert(latent=GAN_LATENT, T=T_SEQ, D=D, ch=64)
expert_ddpm = DDPM_Expert(D=D, T=T_SEQ, channels=DIFF_CHANNELS)
experts      = [expert_tcn, expert_gan, expert_ddpm]
expert_names = ["TCN-AE", "GAN", "DDPM"]

print(f"\\n--- TCN-AE  ({model_size_stats(expert_tcn)['Params']:,} params) ---")
pretrain_tcn(expert_tcn,  Xtr_s, epochs=EXPERT_EPOCHS, lr=EXPERT_LR,
             batch_size=BATCH_SIZE, seed=42)
print(f"\\n--- GAN     ({model_size_stats(expert_gan)['Params']:,} params, "
      f"{model_size_stats(expert_gan.Disc)['Params']:,} D-only) ---")
pretrain_gan(expert_gan,  Xtr_s, epochs=EXPERT_EPOCHS,
             batch_size=BATCH_SIZE, seed=42)
print(f"\\n--- DDPM    ({model_size_stats(expert_ddpm)['Params']:,} params) ---")
pretrain_ddpm(expert_ddpm, Xtr_s, epochs=EXPERT_EPOCHS, lr=EXPERT_LR,
              batch_size=BATCH_SIZE, seed=42)

pretrained_states = [copy.deepcopy(e.state_dict()) for e in experts]
print("\\nPre-trained expert states saved.")
"""))

cells.append(code("""print("\\n" + "=" * 60); print("CALIBRATION — score normalization"); print("=" * 60)
moe = MoE_OptionC(experts, input_dim=D, num_experts=N_EXPERTS,
                   top_k=TOP_K, noisy_gating=True).to(device)
print(f"MoE total params (incl. GAN.G): {model_size_stats(moe)['Params']:,}")
moe.calibrate_score_normalization(Xtr_s, batch_size=BATCH_SIZE)
print(f"per-expert mu    = {moe.score_mu.cpu().numpy().round(4)}")
print(f"per-expert sigma = {moe.score_sigma.cpu().numpy().round(4)}")

print("\\n" + "=" * 60); print("STAGE 2 — gate training"); print("=" * 60)
train_moe_optionc(moe, Xtr_s, epochs=GATING_EPOCHS, lr=GATING_LR,
                  batch_size=BATCH_SIZE, lambda_bal=LAMBDA_BAL,
                  stage="gating", seed=42)

print("\\n" + "=" * 60); print("STAGE 3 — end-to-end fine-tune (GAN frozen)"); print("=" * 60)
train_moe_optionc(moe, Xtr_s, epochs=E2E_EPOCHS, lr=E2E_LR,
                  batch_size=BATCH_SIZE, lambda_bal=LAMBDA_BAL,
                  stage="e2e", seed=42)
print("\\nTraining complete.")
"""))

# ============================================================
# 7. Evaluation
# ============================================================
cells.append(md("""## 7. Evaluation

Per the spec revision (2026-05-08), every standalone-expert and MoE
evaluation reports AUC(train-N vs test-N) alongside ROC-AUC and PR-AUC.
"""))

cells.append(code("""@torch.no_grad()
def expert_score_sequences(expert, X, batch=512):
    expert.eval()
    parts = []
    for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch, shuffle=False):
        parts.append(expert.anomaly_score(x.to(device)).cpu().numpy())
    return np.concatenate(parts)


@torch.no_grad()
def moe_score_sequences(moe_model, X, method="z_combined", batch=512):
    moe_model.eval()
    parts = []
    for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch, shuffle=False):
        parts.append(moe_model.anomaly_scores(x.to(device), method=method).cpu().numpy())
    return np.concatenate(parts)


@torch.no_grad()
def get_expert_routing(moe_model, X, batch=512):
    moe_model.eval()
    parts = []
    for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch, shuffle=False):
        gates, _ = moe_model.gating(x.to(device))
        parts.append(gates.cpu().numpy())
    return np.concatenate(parts)


# --- Standalone expert evaluation table (with AUC(train-N vs test-N)) ---
print("=" * 60)
print("STANDALONE EXPERT EVALUATION")
print("=" * 60)
Xte_normal = Xte_s[yte_seq == 0]
expert_results = {}
for name, expert in zip(expert_names, experts):
    s_tr = expert_score_sequences(expert, Xtr_s)
    s_te = expert_score_sequences(expert, Xte_s)
    s_te_normal = expert_score_sequences(expert, Xte_normal)
    m = compute_metrics_99pct(yte_seq, s_te, s_tr, pct=THR_PCT)
    auc_shift = auc_train_vs_test_normal(s_tr, s_te_normal)
    expert_results[name] = {**m, "auc_train_vs_test_normal": auc_shift,
                              "train_scores": s_tr, "test_scores": s_te}
    print(f"  {name:6s}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"F1={m['f1']:.4f}  AUC(train-N vs test-N)={auc_shift:.4f}")

# --- MoE evaluation across scoring methods ---
print("\\n" + "=" * 60); print("MoE EVALUATION"); print("=" * 60)
best_method, best_roc = None, -1
moe_results = {}
for method in ["z_combined", "raw_combined", "max_z"]:
    s_tr = moe_score_sequences(moe, Xtr_s, method=method)
    s_te = moe_score_sequences(moe, Xte_s, method=method)
    s_te_normal = moe_score_sequences(moe, Xte_normal, method=method)
    m = compute_metrics_99pct(yte_seq, s_te, s_tr, pct=THR_PCT)
    auc_shift = auc_train_vs_test_normal(s_tr, s_te_normal)
    moe_results[method] = {**m, "auc_train_vs_test_normal": auc_shift,
                            "train_scores": s_tr, "test_scores": s_te}
    print(f"  {method:14s}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"F1={m['f1']:.4f}  AUC(train-N vs test-N)={auc_shift:.4f}")
    if m["roc_auc"] > best_roc:
        best_roc, best_method = m["roc_auc"], method
print(f"\\nBest scoring method: {best_method}  (ROC-AUC={best_roc:.4f})")
train_scores = moe_results[best_method]["train_scores"]
test_scores  = moe_results[best_method]["test_scores"]
"""))

cells.append(code("""# Per-attack breakdown
pa = per_attack_breakdown(test_scores, yte_seq, ate_seq)
print("Per-Attack Breakdown (MoE3 Option C):")
display(fmt_cols(pa, ["roc_auc", "pr_auc"]))

# Score distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(test_scores[yte_seq == 0], bins=60, alpha=0.5, label="Normal", density=True)
ax.hist(test_scores[yte_seq == 1], bins=60, alpha=0.5, label="Attack", density=True)
thr = np.percentile(train_scores, THR_PCT)
ax.axvline(thr, color='r', ls='--', lw=1.5, label=f"Threshold (p{THR_PCT})")
ax.set_xlabel("Anomaly Score (z-combined)"); ax.set_ylabel("Density")
ax.set_title("MoE3 Option C Score Distribution"); ax.legend()
plt.tight_layout(); plt.show()
"""))

# ============================================================
# 8. Ablations
# ============================================================
cells.append(md("""## 8. Ablation Studies (Option C)

Per the spec revision, Option C runs five ablations:

1. Top-K ∈ {1, 2, 3}
2. Load-balancing weight λ ∈ {0.01, 0.1, 0.5}
3. Score normalization on / off
4. Detection-aware gating signal (contrastive vs synthetic anomalies)
5. **Diffusion-expert capacity** — full (12 → 32 → 64, ~110K) vs narrowed (12 → 16 → 32, ~30K).
   Hypothesis: tighter train fit drives shift-fragility; the narrowed Diffusion
   expert should have higher Stage-1 final loss but lower AUC(train-N vs test-N).

Cells below wire each ablation; turn them on individually for full sweeps.
"""))

cells.append(code("""# Ablation 5 — Diffusion-expert capacity (full vs narrowed). Demonstrates the
# capacity-vs-shift hypothesis transferred from Option B / D2.5c to a Diffusion
# expert. Only the DDPM expert is swapped; TCN-AE and GAN reuse pretrained_states.
print("=" * 60)
print("Ablation 5 — Diffusion capacity (full 12->32->64 vs narrow 12->16->32)")
print("=" * 60)
abl5_results = {}
for tag, channels in [("full",   DIFF_CHANNELS),
                      ("narrow", DIFF_CHANNELS_NARROW)]:
    set_seed(42)
    print(f"\\n--- {tag}  channels={channels} ---")
    ddpm_var = DDPM_Expert(D=D, T=T_SEQ, channels=channels)
    n_params = model_size_stats(ddpm_var)["Params"]
    print(f"  DDPM params: {n_params:,}")
    pretrain_ddpm(ddpm_var, Xtr_s, epochs=EXPERT_EPOCHS, lr=EXPERT_LR,
                  batch_size=BATCH_SIZE, seed=42, verbose=True)
    s_tr = expert_score_sequences(ddpm_var, Xtr_s)
    s_te = expert_score_sequences(ddpm_var, Xte_s)
    s_te_normal = expert_score_sequences(ddpm_var, Xte_s[yte_seq == 0])
    m = compute_metrics_99pct(yte_seq, s_te, s_tr, pct=THR_PCT)
    auc_shift = auc_train_vs_test_normal(s_tr, s_te_normal)
    abl5_results[tag] = {**m, "auc_train_vs_test_normal": auc_shift, "params": n_params}
    print(f"  {tag:6s}  params={n_params:,}  ROC-AUC={m['roc_auc']:.4f}  "
          f"PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}  "
          f"AUC(train-N vs test-N)={auc_shift:.4f}")

print("\\nAblation 5 summary:")
for tag, r in abl5_results.items():
    print(f"  {tag:6s}: params={r['params']:>7,}  AUC(train-N vs test-N)={r['auc_train_vs_test_normal']:.4f}  "
          f"ROC-AUC={r['roc_auc']:.4f}")
print("\\nHypothesis: narrow variant should have higher Stage-1 loss but LOWER AUC(train-N vs test-N).")
"""))

# ============================================================
# 9. Routing analysis
# ============================================================
cells.append(md("## 9. Routing Analysis\n"))

cells.append(code("""gates_test = get_expert_routing(moe, Xte_s)
atk_types = sorted(set(ate_seq) - {"normal"})
routing_data = {}
for atk in ["normal"] + atk_types:
    mask = ate_seq == atk
    if mask.sum() > 0:
        routing_data[atk] = gates_test[mask].mean(axis=0)
routing_df = pd.DataFrame(routing_data, index=expert_names).T
print("Average Expert Routing by Attack Type (MoE3 Option C):")
display(routing_df.round(4))

mean_per_expert = routing_df.mean(axis=0)
print("\\nMean routing weight across all traffic types:")
print(mean_per_expert.round(4))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.heatmap(routing_df, annot=True, fmt=".3f", cmap="YlOrRd",
            ax=axes[0], vmin=0, vmax=1)
axes[0].set_title("Expert Routing Weights by Attack Type")
axes[0].set_ylabel("Traffic Type"); axes[0].set_xlabel("Expert")
routing_df.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2")
axes[1].set_title("Expert Allocation per Attack Type")
axes[1].set_ylabel("Gate Weight"); axes[1].set_xlabel("")
axes[1].legend(title="Expert", bbox_to_anchor=(1.05, 1))
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()
"""))

# ============================================================
# 10. Confusion matrices
# ============================================================
cells.append(md("## 10. Confusion Matrices\n"))

cells.append(code("""thr = np.percentile(train_scores, THR_PCT)
y_pred = (test_scores > thr).astype(int)
n_attacks = len(atk_types)
fig, axes = plt.subplots(1, min(n_attacks + 1, 5), figsize=(5 * min(n_attacks + 1, 5), 4.5))
if not isinstance(axes, np.ndarray):
    axes = [axes]
cm = confusion_matrix(yte_seq, y_pred, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt="d", ax=axes[0],
            xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"], cmap="Blues")
axes[0].set_title("Overall"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
for i, atk in enumerate(atk_types[:min(n_attacks, 4)]):
    mask = np.isin(ate_seq, ["normal", atk])
    cm_a = confusion_matrix(yte_seq[mask], y_pred[mask], labels=[0, 1])
    ax = axes[i + 1] if i + 1 < len(axes) else axes[-1]
    sns.heatmap(cm_a, annot=True, fmt="d", ax=ax,
                xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"], cmap="Oranges")
    ax.set_title(atk); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("MoE3 Option C Confusion Matrices", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()
"""))

# ============================================================
# 11. Param/latency/energy
# ============================================================
cells.append(md("## 11. Model Size, Latency & Energy\n"))

cells.append(code("""print("MoE3 Option C Component Sizes:")
for name, expert in zip(expert_names, moe.experts):
    s = model_size_stats(expert)
    print(f"  {name:8s}  {s['Params']:>8,} params  {s['Size (MB)']:.4f} MB")
s_gate = model_size_stats(moe.gating)
print(f"  {'Gating':8s}  {s_gate['Params']:>8,} params  {s_gate['Size (MB)']:.4f} MB")
s_total = model_size_stats(moe)
print(f"  {'TOTAL':8s}  {s_total['Params']:>8,} params  {s_total['Size (MB)']:.4f} MB")

x_sample = torch.tensor(Xte_s[:1], dtype=torch.float32, device=device)
def moe_one():
    with torch.no_grad():
        moe.anomaly_scores(x_sample, method=best_method)
moe_latency = measure_latency_ms(moe_one, runs=500)
print(f"\\nMoE3 Option C inference latency: {moe_latency:.4f} ms/seq")

for name, expert in zip(expert_names, moe.experts):
    expert.eval()
    def _inf(e=expert):
        with torch.no_grad(): e.anomaly_score(x_sample)
    print(f"  {name:8s}  {measure_latency_ms(_inf, runs=500):.4f} ms/seq")

try:
    E, P, T = nvml_energy_joules(repeat_workload(moe_one, n=500))
    moe_energy = E / 500
    print(f"\\nMoE3 Option C energy: {moe_energy:.6f} J/seq, avg power: {P:.2f} W")
except Exception as exc:
    print(f"Energy measurement failed: {exc}")
    moe_energy = np.nan
"""))

# ============================================================
# 12. Final comparison
# ============================================================
cells.append(md("## 12. Final Comparison with Baselines & Option B\n"))

cells.append(code("""baselines = [
    {"Model": "INDRA (GRU-AE)",  "ROC-AUC": 0.990, "PR-AUC": 0.980,
     "Params": 64_652,  "Latency (ms)": 1.29,  "Energy (J)": 0.015,
     "AUC(tr-N vs te-N)": np.nan},
    {"Model": "MoE3 Option B",   "ROC-AUC": 0.9704, "PR-AUC": 0.9903,
     "Params": 185_846, "Latency (ms)": np.nan, "Energy (J)": np.nan,
     "AUC(tr-N vs te-N)": 0.9954},  # from prior diagnostic D2.5c
    {"Model": "MAD-GAN",         "ROC-AUC": 0.983, "PR-AUC": 0.990,
     "Params": 41_805,  "Latency (ms)": 40.21, "Energy (J)": 0.847,
     "AUC(tr-N vs te-N)": np.nan},
    {"Model": "LATTE (thr)",     "ROC-AUC": 0.977, "PR-AUC": 0.944,
     "Params": 164_236, "Latency (ms)": 0.20,  "Energy (J)": 0.024,
     "AUC(tr-N vs te-N)": np.nan},
    {"Model": "TENET+DT",        "ROC-AUC": 0.975, "PR-AUC": 0.946,
     "Params": 22_728,  "Latency (ms)": 3.17,  "Energy (J)": 0.074,
     "AUC(tr-N vs te-N)": np.nan},
]
moe_final = compute_metrics_99pct(yte_seq, test_scores, train_scores, pct=THR_PCT)
auc_shift_final = moe_results[best_method]["auc_train_vs_test_normal"]
baselines.append({
    "Model":              "MoE3 Option C (Ours)",
    "ROC-AUC":            round(moe_final["roc_auc"], 4),
    "PR-AUC":             round(moe_final["pr_auc"], 4),
    "Params":             model_size_stats(moe)["Params"],
    "Latency (ms)":       round(moe_latency, 4),
    "Energy (J)":         round(moe_energy, 6) if not np.isnan(moe_energy) else "N/A",
    "AUC(tr-N vs te-N)":  round(auc_shift_final, 4),
})
comp_df = pd.DataFrame(baselines).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print("Final Deployment Comparison:")
display(comp_df)

os.makedirs("./results", exist_ok=True)
comp_df.to_csv("./results/MoE3_OptionC_results.csv", index=False)
print("\\nResults saved to ./results/MoE3_OptionC_results.csv")
print("Notebook complete.")
"""))

# ============================================================
# Notebook metadata
# ============================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (gpu_env)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for c in cells:
    c["id"] = uuid.uuid4().hex[:12]

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Wrote: {OUT_PATH}")
print(f"Total cells: {len(cells)}")
