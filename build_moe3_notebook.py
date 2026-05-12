"""Build MoE3-IDS-OptionB.ipynb based on TOW-IDS-MoE.ipynb structure.

Reuses data pipeline, gating network, MoE wrapper, training loop, and evaluation
harness verbatim. Replaces the 4-expert list with 3 experts: GRU-AE, TCN-AE, GAT-AE.
The GAT-AE expert (with custom MultiHeadGraphAttention) is the new contribution.
"""
import json
import os

OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MoE3-IDS-OptionB.ipynb",
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

# ------------------------------------------------------------------
# 0. Title
# ------------------------------------------------------------------
cells.append(md("""# MoE3-IDS Option B — Three-Expert MoE for Automotive Ethernet IDS

Three-expert Mixture-of-Experts (MoE) anomaly detection on the TOW-IDS dataset
(Han et al., IEEE TIFS 2023). Implements **Option B** of `MoE3_design_spec.md`:
cross-axis expert diversity to address the expert-collapse problem observed in
the 4-expert MoE-IDS.

**Experts:**
| # | Expert | Operates over | Captures |
|---|--------|---------------|----------|
| 1 | GRU-AE  | time axis (recurrent)         | sequential state evolution |
| 2 | TCN-AE  | time axis (multi-scale conv)  | periodic patterns at scales 1, 2, 4 |
| 3 | GAT-AE  | feature axis (graph attention) | cross-feature correlations |

**Training Strategy** (verbatim from TOW-IDS-MoE):
1. Stage 1 — Pre-train each expert independently on normal traffic
2. Stage 2 — Train the gating network (experts frozen)
3. Stage 3 — End-to-end fine-tuning (all parameters, low LR)

**Evaluation:** ROC-AUC, PR-AUC, F1, per-attack breakdown, ablation studies,
expert routing analysis, confusion matrices, latency/energy profiling.
"""))

# ------------------------------------------------------------------
# 1. Imports (verbatim from TOW-IDS-MoE)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 2. File paths + MoE hyperparameters (3 experts now)
# ------------------------------------------------------------------
cells.append(code("""# === File paths (verbatim from TOW-IDS-MoE) ===
TRAIN_PCAP = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap"
TEST_PCAP  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap"
OUT_DIR    = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/extracted"
y_train_path = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_train.csv"
y_test_path  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_test.csv"

# === Sequence / evaluation defaults ===
WINDOW_SEC = 1.0
T_SEQ      = 40
THR_PCT    = 99

# === MoE hyperparameters — Option B (3 experts) ===
N_EXPERTS      = 3
TOP_K          = 2
LAMBDA_BAL     = 0.01   # load-balancing loss weight
EXPERT_EPOCHS  = 100
GATING_EPOCHS  = 50
E2E_EPOCHS     = 30
EXPERT_LR      = 1e-3
GATING_LR      = 1e-3
E2E_LR         = 1e-4
BATCH_SIZE     = 256

# === GAT-AE hyperparameters (Option B novel expert) ===
GAT_HIDDEN     = 32
GAT_HEADS      = 2
GAT_TOPK       = 4   # feature-graph top-K (ablate over {2, 4, 6, 12})
GAT_BOTTLENECK = 16
"""))

# ------------------------------------------------------------------
# 3. Utility functions (verbatim from TOW-IDS-MoE)
# ------------------------------------------------------------------
cells.append(code("""# =============================================
# Utility Functions (verbatim from TOW-IDS-MoE)
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


def combined_model_size(*models):
    params = sum(p.numel() for m in models for p in m.parameters())
    bytes_ = sum(p.numel() * p.element_size() for m in models for p in m.parameters())
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


def per_attack_breakdown(scores, y_seq, atk_seq):
    df = pd.DataFrame({"score": scores, "y": y_seq.astype(int), "atk": atk_seq.astype(str)})
    attacks = sorted([a for a in df["atk"].unique() if a != "normal"])
    rows = []
    for a in attacks:
        sub = df[df["atk"].isin(["normal", a])]
        if sub["y"].nunique() < 2:
            continue
        rows.append({
            "attack": a,
            "n_attack_seq": int((sub["atk"] == a).sum()),
            "roc_auc": roc_auc_score(sub["y"], sub["score"]),
            "pr_auc":  average_precision_score(sub["y"], sub["score"]),
        })
    return pd.DataFrame(rows)


def seq_scores_to_window_scores(seq_scores, n_windows, t_seq=40, stride=1, agg="mean"):
    buckets = [[] for _ in range(n_windows)]
    for i, s in enumerate(seq_scores):
        start = i * stride
        for j in range(t_seq):
            w = start + j
            if w < n_windows:
                buckets[w].append(s)
    agg_fn = np.mean if agg == "mean" else np.max
    return np.array([agg_fn(b) if b else 0.0 for b in buckets])


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


print("Utility functions loaded.")
"""))

# ------------------------------------------------------------------
# Section 1. Data Ingestion (verbatim)
# ------------------------------------------------------------------
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
print(f"Train packets: {train_pkts.shape}  |  Test packets: {test_pkts.shape}")

# --- Load & merge labels ---
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

# --- Normalize attack_type (handle merge-suffix cases) ---
for df in [train_pkts, test_pkts]:
    at_col = None
    for c in ["attack_type", "attack_type_y", "attack_type_x"]:
        if c in df.columns and df[c].notna().any():
            at_col = c
            break
    if at_col and at_col != "attack_type":
        df["attack_type"] = df[at_col]
    df["attack_type"] = df["attack_type"].astype(str).str.strip().str.lower()

print("Train labels:", train_pkts["y"].value_counts().to_dict())
print("Test  labels:", test_pkts["y"].value_counts().to_dict())
print("Test attack types:\\n", test_pkts.loc[test_pkts["y"]==1, "attack_type"].value_counts().head(10))
"""))

# ------------------------------------------------------------------
# Section 2. Feature Engineering (verbatim)
# ------------------------------------------------------------------
cells.append(md("""## 2. Feature Engineering

12 windowed features (1-second windows) + sliding-window sequences of length T_SEQ=40.
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
    X   = df[feature_cols].astype(float).to_numpy()
    y   = df["y_window"].astype(int).to_numpy()
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
    scaler = StandardScaler()
    scaler.fit(Xtr_normal.reshape(-1, D))
    def sx(X):
        n, t, d = X.shape
        return scaler.transform(X.reshape(-1, d)).reshape(n, t, d)
    return sx(Xtr_normal), sx(Xte_all), scaler


print("Feature functions defined.")
"""))

cells.append(code("""# Build 1-second windowed features
train_w = build_global_windows_with_attack(train_pkts, window_sec=WINDOW_SEC)
test_w  = build_global_windows_with_attack(test_pkts,  window_sec=WINDOW_SEC)
print(f"Train windows: {train_w.shape}  |  Test windows: {test_w.shape}")
print(f"Train: normal={int((train_w['y_window']==0).sum())}, attack={int((train_w['y_window']==1).sum())}")
print(f"Test attack types:\\n{test_w.loc[test_w['y_window']==1, 'attack_type_window'].value_counts()}")

# Build sliding-window sequences
feat_cols = get_feature_cols(train_w)
D = len(feat_cols)
print(f"\\nFeature dim D={D}, Sequence length T_SEQ={T_SEQ}")

Xtr_seq, ytr_seq, atr_seq = make_sequences(
    train_w[train_w["y_window"] == 0].reset_index(drop=True),
    t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
Xte_seq, yte_seq, ate_seq = make_sequences(
    test_w, t_seq=T_SEQ, stride=1, feature_cols=feat_cols)

# StandardScaler fit on normal-train, transform both
Xtr_s, Xte_s, seq_scaler = scale_train_test_sequences(Xtr_seq, Xte_seq)
Xtr_s = Xtr_s.astype(np.float32)
Xte_s = Xte_s.astype(np.float32)

print(f"Train sequences (normal): {Xtr_s.shape}")
print(f"Test  sequences (all):    {Xte_s.shape}")
print(f"Test labels - normal: {int((yte_seq==0).sum())}, anomaly: {int((yte_seq==1).sum())}")
"""))

# ------------------------------------------------------------------
# Section 3. EDA (verbatim)
# ------------------------------------------------------------------
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

# Quick sequence-level check
print(f"Xtr_s range: [{Xtr_s.min():.2f}, {Xtr_s.max():.2f}]")
print(f"Xte_s range: [{Xte_s.min():.2f}, {Xte_s.max():.2f}]")
"""))

# ------------------------------------------------------------------
# Section 4. Expert Architectures (3 experts)
# ------------------------------------------------------------------
cells.append(md("""## 4. Expert Architectures (Option B: cross-axis diversity)

Three experts, each operating on a fundamentally different axis of the input tensor.
The structural specialization is the source of diversity rather than learned routing —
addressing the expert-collapse problem of the 4-expert MoE-IDS.

| # | Expert | Architecture | Operates over | Inductive Bias |
|---|--------|-------------|---------------|----------------|
| 1 | **GRU-AE** | GRU encoder-decoder            | time axis (recurrent)         | Sequential temporal patterns |
| 2 | **TCN-AE** | Dilated causal Conv1d AE       | time axis (multi-scale conv)  | Periodic patterns at scales 1, 2, 4 |
| 3 | **GAT-AE** | Graph attention over features  | feature axis (graph)          | Cross-feature correlations |

All experts share the same interface: input `(B, T, D)` → reconstruction `(B, T, D)`.
Anomaly score = MSE between input and reconstruction.

GRU-AE and TCN-AE are copied verbatim from `TOW-IDS-MoE.ipynb`. GAT-AE is the new
contribution introduced by this Option B experiment.
"""))

# Reused experts (verbatim)
cells.append(code("""# ============================================================
# Reused experts (verbatim from TOW-IDS-MoE.ipynb)
# ============================================================
class GRU_AE_Expert(nn.Module):
    'Expert 1: GRU Autoencoder (INDRA-style). Captures sequential temporal patterns.'
    def __init__(self, D, hidden=64, proj=128, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(D, proj), nn.Tanh())
        self.drop = nn.Dropout(dropout)
        self.enc = nn.GRU(proj, hidden, batch_first=True)
        self.dec = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, D)

    def forward(self, x):
        z = self.drop(self.in_proj(x))
        enc_out, _ = self.enc(z)
        dec_out, _ = self.dec(self.drop(enc_out))
        return self.out(self.drop(dec_out))


class TCN_AE_Expert(nn.Module):
    'Expert 2: Temporal CNN Autoencoder (TENET-style). Multi-scale temporal patterns.'
    def __init__(self, D, channels=64, kernel_size=3):
        super().__init__()
        k = kernel_size
        # Encoder - dilated convolutions (dilation 1, 2, 4)
        self.enc1 = nn.Conv1d(D, channels,   k, padding=(k-1)*1//2, dilation=1)
        self.enc2 = nn.Conv1d(channels, channels*2, k, padding=(k-1)*2//2, dilation=2)
        self.enc3 = nn.Conv1d(channels*2, channels,  k, padding=(k-1)*4//2, dilation=4)
        # Decoder - standard convolutions
        self.dec1 = nn.Conv1d(channels, channels*2, k, padding=(k-1)//2)
        self.dec2 = nn.Conv1d(channels*2, channels,  k, padding=(k-1)//2)
        self.dec3 = nn.Conv1d(channels, D, 1)

    def forward(self, x):
        h = x.transpose(1, 2)                       # (B, D, T)
        h = F.relu(self.enc1(h))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        h = self.dec3(h)
        return h.transpose(1, 2)                     # (B, T, D)


print("GRU_AE_Expert and TCN_AE_Expert defined (verbatim from TOW-IDS-MoE).")
"""))

# NEW: GAT-AE Expert
cells.append(code("""# ============================================================
# NEW: MultiHeadGraphAttention + GAT_AE_Expert (Option B novelty)
# ============================================================
class MultiHeadGraphAttention(nn.Module):
    \"\"\"Custom multi-head graph attention layer.

    Per the MoE3 design spec, attention weights are supplied externally via the
    pre-computed adjacency matrix `adj` (top-K cosine-similarity over feature nodes).
    Each head linearly projects the input then propagates with `adj`, the per-head
    outputs are concatenated and projected back to `out_dim`.

        per-head:        h_h = adj @ (W_h @ x)        in R^{B x N x out_dim}
        concatenate:     h   = concat_h h_h           in R^{B x N x heads*out_dim}
        project:         y   = W_o @ h                in R^{B x N x out_dim}

    No PyTorch Geometric dependency.
    \"\"\"
    def __init__(self, in_dim, out_dim, heads=2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        # All heads in one projection: in_dim -> heads * out_dim
        self.W = nn.Linear(in_dim, heads * out_dim)
        # Post-concat projection back to out_dim
        self.proj = nn.Linear(heads * out_dim, out_dim)

    def forward(self, x, adj):
        # x:   (B, N, in_dim)
        # adj: (B, N, N)
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.out_dim)   # (B, N, H, out_dim)
        outs = []
        for i in range(self.heads):
            outs.append(torch.bmm(adj, h[:, :, i, :]))         # (B, N, out_dim)
        out = torch.cat(outs, dim=-1)                          # (B, N, H*out_dim)
        return self.proj(out)                                   # (B, N, out_dim)


class GAT_AE_Expert(nn.Module):
    \"\"\"Expert 3: Graph Attention Autoencoder over the feature axis.

    Builds a dynamic top-K cosine-similarity graph over the D=12 features (treated
    as graph nodes) per window. Captures cross-feature dependencies that are
    orthogonal to the temporal patterns of GRU-AE and TCN-AE — the core hypothesis
    of the cross-axis-diversity Option B experiment.

    Input  shape: (B, T, D)  with T=40, D=12
    Output shape: (B, T, D)  reconstruction
    \"\"\"
    def __init__(self, D=12, T=40, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                 top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK):
        super().__init__()
        self.D, self.T = D, T
        self.top_k = min(top_k, D)
        # Encoder: lift T -> hidden, then 2x graph-attention layers down to bottleneck
        self.enc_lin1  = nn.Linear(T, hidden)
        self.enc_attn1 = MultiHeadGraphAttention(hidden, hidden,    heads=heads)
        self.enc_attn2 = MultiHeadGraphAttention(hidden, bottleneck, heads=heads)
        # Decoder: 2x graph-attention layers back up to hidden, then hidden -> T
        self.dec_attn1 = MultiHeadGraphAttention(bottleneck, hidden, heads=heads)
        self.dec_attn2 = MultiHeadGraphAttention(hidden,     hidden, heads=heads)
        self.dec_lin   = nn.Linear(hidden, T)

    def build_dynamic_adj(self, x_feat):
        \"\"\"Top-K cosine-similarity adjacency over feature nodes.

        x_feat: (B, D, T)  -> adj: (B, D, D)
        \"\"\"
        x_norm = F.normalize(x_feat, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))      # (B, D, D)
        topk_vals, topk_idx = sim.topk(self.top_k, dim=-1)   # keep K most similar peers
        adj = torch.zeros_like(sim)
        adj.scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))
        return adj

    def forward(self, x):
        # (B, T, D) -> (B, D, T) so feature axis acts as the graph node axis
        x_feat = x.transpose(1, 2)
        adj = self.build_dynamic_adj(x_feat)
        h = F.elu(self.enc_lin1(x_feat))         # (B, D, hidden)
        h = F.elu(self.enc_attn1(h, adj))         # (B, D, hidden)
        z = self.enc_attn2(h, adj)                # (B, D, bottleneck)
        h = F.elu(self.dec_attn1(z, adj))         # (B, D, hidden)
        h = F.elu(self.dec_attn2(h, adj))         # (B, D, hidden)
        x_recon = self.dec_lin(h)                 # (B, D, T)
        return x_recon.transpose(1, 2)            # (B, T, D)


print("MultiHeadGraphAttention and GAT_AE_Expert defined.")
for cls, kw in [(GRU_AE_Expert, dict(D=12)),
                (TCN_AE_Expert, dict(D=12)),
                (GAT_AE_Expert, dict(D=12, T=40))]:
    tmp = cls(**kw)
    print(f"  {cls.__name__:20s} - {model_size_stats(tmp)['Params']:>8,} params")
"""))

# ------------------------------------------------------------------
# Section 5. Sparse Gating Network & MoE Framework (verbatim)
# ------------------------------------------------------------------
cells.append(md("""## 5. Sparse Gating Network & MoE Framework (verbatim)

**Gating:** Temporal mean+std pooling → 2-layer MLP → sparse top-K softmax.
Noisy gating (learned noise std) encourages exploration during training.

**Load-Balancing Loss** (Fedus et al., 2021):
$\\mathcal{L}_{\\text{bal}} = N \\cdot \\sum_{i=1}^{N} f_i \\cdot P_i$
where $f_i$ = avg gate value for expert $i$, $P_i$ = avg routing probability.
"""))

cells.append(code("""class SparseGatingNetwork(nn.Module):
    'Sparse Top-K Gating with Noisy Routing (Shazeer et al., 2017).'
    def __init__(self, seq_dim, num_experts, top_k=2, noisy=True):
        super().__init__()
        input_dim = seq_dim * 2          # mean + std pooling
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy = noisy
        self.gate = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                  nn.Linear(64, num_experts))
        if noisy:
            self.noise_linear = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                              nn.Linear(64, num_experts))

    def forward(self, x):
        x_agg = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)   # (B, 2D)
        logits = self.gate(x_agg)                                    # (B, E)
        if self.noisy and self.training:
            noise_std = F.softplus(self.noise_linear(x_agg))
            logits = logits + torch.randn_like(logits) * noise_std
        # Sparse top-K
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        sparse = torch.full_like(logits, float('-inf'))
        sparse.scatter_(1, top_idx, top_vals)
        gates = F.softmax(sparse, dim=-1)                            # (B, E)
        return gates, logits


class MoE_IDS(nn.Module):
    'Mixture-of-Experts IDS: sparse gating over expert autoencoders.'
    def __init__(self, experts, input_dim, num_experts=4, top_k=2, noisy_gating=True):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating = SparseGatingNetwork(input_dim, num_experts, top_k, noisy_gating)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        gates, raw_logits = self.gating(x)                           # (B, E)
        recons = torch.stack([e(x) for e in self.experts], dim=1)    # (B, E, T, D)
        g = gates.unsqueeze(-1).unsqueeze(-1)                        # (B, E, 1, 1)
        output = (g * recons).sum(dim=1)                             # (B, T, D)
        return output, gates, raw_logits, recons

    def anomaly_scores(self, x, method='combined_mse'):
        output, gates, _, recons = self.forward(x)
        if method == 'combined_mse':
            return torch.mean((x - output) ** 2, dim=(1, 2))
        elif method == 'weighted_expert_mse':
            emse = torch.mean((x.unsqueeze(1) - recons) ** 2, dim=(2, 3))  # (B, E)
            return (gates * emse).sum(dim=1)
        else:
            return torch.mean((x - output) ** 2, dim=(1, 2))


def load_balancing_loss(gates, raw_logits, num_experts):
    'Auxiliary loss encouraging balanced expert utilisation.'
    f = gates.mean(dim=0)
    P = F.softmax(raw_logits, dim=-1).mean(dim=0)
    return num_experts * (f * P).sum()


print("MoE framework defined.")
"""))

# ------------------------------------------------------------------
# Section 6. Training Pipeline (verbatim)
# ------------------------------------------------------------------
cells.append(md("""## 6. Training Pipeline (verbatim)

| Stage | What is trained | Experts | Epochs | LR |
|-------|----------------|---------|--------|----|
| 1 | Each expert independently | Unfrozen | 100 | 1e-3 |
| 2 | Gating network only | Frozen | 50 | 1e-3 |
| 3 | Everything end-to-end | Unfrozen | 30 | 1e-4 |
"""))

cells.append(code("""def pretrain_expert(expert, Xtr_s, *, epochs=100, lr=1e-3, batch_size=256, seed=0):
    'Stage 1: Pre-train one expert autoencoder on normal sequences.'
    set_seed(seed)
    model = expert.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        losses = []
        for (x,) in loader:
            x = x.to(device)
            loss = loss_fn(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    [{type(model).__name__}] ep {ep:3d}: loss={np.mean(losses):.6f}")
    return model


def train_moe(moe_model, Xtr_s, *, epochs=50, lr=1e-4, batch_size=256,
              lambda_bal=0.01, freeze_experts=True, seed=0):
    'Train MoE: Stage 2 (freeze_experts=True) or Stage 3 (False).'
    set_seed(seed)
    moe_model = moe_model.to(device)

    if freeze_experts:
        for e in moe_model.experts:
            for p in e.parameters(): p.requires_grad = False
        params = list(moe_model.gating.parameters())
    else:
        for e in moe_model.experts:
            for p in e.parameters(): p.requires_grad = True
        params = list(moe_model.parameters())

    loader = DataLoader(TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()
    moe_model.train()

    for ep in range(epochs):
        ep_rec, ep_bal = [], []
        for (x,) in loader:
            x = x.to(device)
            output, gates, raw_logits, _ = moe_model(x)
            rec = loss_fn(output, x)
            bal = load_balancing_loss(gates, raw_logits, moe_model.num_experts)
            loss = rec + lambda_bal * bal
            opt.zero_grad(); loss.backward(); opt.step()
            ep_rec.append(rec.item()); ep_bal.append(bal.item())
        if ep % 10 == 0 or ep == epochs - 1:
            tag = "Gating" if freeze_experts else "E2E"
            print(f"    [MoE-{tag}] ep {ep:3d}: rec={np.mean(ep_rec):.6f}  bal={np.mean(ep_bal):.4f}")
    return moe_model


print("Training functions defined.")
"""))

# ------------------------------------------------------------------
# Section 6.5: Smoke Test (NEW per spec)
# ------------------------------------------------------------------
cells.append(md("""## 6.5 Smoke Test (1 epoch, 100 samples) — required by spec

Per `MoE3_design_spec.md` § Smoke Test: before launching Stage 1 training, run a
1-epoch dry pass on a 100-sample subset to catch shape errors and confirm that
all three experts produce `(B, 40, 12)` reconstructions. Print parameter counts
for each expert and the total. **If parameter counts deviate from the spec by
more than 10%, stop and review.**
"""))

cells.append(code("""# ============================================================
# Smoke Test — 1 epoch, 100 samples, all three stages
# ============================================================
print("=" * 60)
print("SMOKE TEST — 1 epoch, 100 samples, all three stages")
print("=" * 60)

set_seed(42)
SMOKE_N      = 100
SMOKE_EPOCHS = 1
SMOKE_BATCH  = 32

# --- 1. Param counts vs spec target ---
spec_params = {"GRU-AE": 65_000, "TCN-AE": 102_000, "GAT-AE": 35_000}
smoke_experts = [
    GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2),
    TCN_AE_Expert(D, channels=64, kernel_size=3),
    GAT_AE_Expert(D=D, T=T_SEQ, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                  top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK),
]
smoke_names = ["GRU-AE", "TCN-AE", "GAT-AE"]

print("\\n[1] Parameter counts vs spec target:")
deviations = []
total = 0
for name, exp in zip(smoke_names, smoke_experts):
    p = model_size_stats(exp)["Params"]
    total += p
    target = spec_params[name]
    dev_pct = 100.0 * (p - target) / target
    flag = "  OK  " if abs(dev_pct) <= 10.0 else " WARN "
    print(f"  [{flag}] {name:7s}  {p:>8,} params   "
          f"(spec target {target:>7,}, deviation {dev_pct:+.1f}%)")
    deviations.append((name, p, target, dev_pct))
print(f"  ----  TOTAL    {total:>8,} params   (spec target ~202K, GAT+gate ~39K)")

# --- 2. Shape check: each expert (B, T, D) -> (B, T, D) ---
print("\\n[2] Shape check on a (4, 40, 12) batch:")
x_dummy = torch.randn(4, T_SEQ, D, device=device)
for name, exp in zip(smoke_names, smoke_experts):
    exp = exp.to(device).eval()
    with torch.no_grad():
        y_dummy = exp(x_dummy)
    ok = tuple(y_dummy.shape) == (4, T_SEQ, D)
    print(f"  [{('OK' if ok else 'FAIL'):4s}] {name:7s}  in={tuple(x_dummy.shape)}  out={tuple(y_dummy.shape)}")
    if not ok:
        raise RuntimeError(f"{name} reconstruction shape mismatch")

# --- 3. 1-epoch Stage 1 pre-training on 100 samples ---
Xtr_smoke = Xtr_s[:SMOKE_N]
print(f"\\n[3] Stage 1: pre-train each expert for {SMOKE_EPOCHS} epoch on {SMOKE_N} samples")
for name, exp in zip(smoke_names, smoke_experts):
    print(f"  --- {name} ---")
    pretrain_expert(exp, Xtr_smoke, epochs=SMOKE_EPOCHS, lr=EXPERT_LR,
                    batch_size=SMOKE_BATCH, seed=42)

# --- 4. 1-epoch Stage 2 gating training (experts frozen) ---
print(f"\\n[4] Stage 2: gating-only training for {SMOKE_EPOCHS} epoch")
moe_smoke = MoE_IDS(smoke_experts, input_dim=D, num_experts=N_EXPERTS,
                    top_k=TOP_K, noisy_gating=True)
print(f"  MoE total params: {model_size_stats(moe_smoke)['Params']:,}")
train_moe(moe_smoke, Xtr_smoke, epochs=SMOKE_EPOCHS, lr=GATING_LR,
          batch_size=SMOKE_BATCH, lambda_bal=LAMBDA_BAL,
          freeze_experts=True, seed=42)

# --- 5. 1-epoch Stage 3 end-to-end fine-tuning ---
print(f"\\n[5] Stage 3: end-to-end fine-tuning for {SMOKE_EPOCHS} epoch")
train_moe(moe_smoke, Xtr_smoke, epochs=SMOKE_EPOCHS, lr=E2E_LR,
          batch_size=SMOKE_BATCH, lambda_bal=LAMBDA_BAL,
          freeze_experts=False, seed=42)

# --- 6. End-to-end MoE forward + scoring shape check ---
print("\\n[6] End-to-end MoE forward shape check:")
moe_smoke.eval()
with torch.no_grad():
    x_check = torch.tensor(Xtr_smoke[:8], dtype=torch.float32, device=device)
    out, gates, raw_logits, recons = moe_smoke(x_check)
    scores_cmb = moe_smoke.anomaly_scores(x_check, method="combined_mse")
    scores_wtd = moe_smoke.anomaly_scores(x_check, method="weighted_expert_mse")
print(f"  out         shape = {tuple(out.shape)}        (expect (8, 40, 12))")
print(f"  gates       shape = {tuple(gates.shape)}              (expect (8, 3))")
print(f"  recons      shape = {tuple(recons.shape)}     (expect (8, 3, 40, 12))")
print(f"  combined_mse       = {tuple(scores_cmb.shape)}                 (expect (8,))")
print(f"  weighted_expert_mse= {tuple(scores_wtd.shape)}                 (expect (8,))")
print(f"  gate row sums   = {gates.sum(dim=-1).cpu().numpy().round(4)} (each should be ~1.0)")

# --- 7. Verdict ---
print("\\n" + "=" * 60)
warns = [d for d in deviations if abs(d[3]) > 10.0]
if warns:
    print(f"SMOKE TEST FINISHED with {len(warns)} parameter-count deviation(s) >10%:")
    for name, p, target, dev in warns:
        print(f"   - {name}: {p:,} vs target {target:,} ({dev:+.1f}%)")
    print("Per spec: review before launching full training.")
else:
    print("SMOKE TEST PASSED. All shapes correct, all params within 10% of spec.")
print("=" * 60)
"""))

# ------------------------------------------------------------------
# Stage 1, 2, 3 (real run cells, gated by user after smoke test)
# ------------------------------------------------------------------
cells.append(md("""---
**STOP — review the smoke-test output above before running the cells below.**

If parameter counts and shape checks look correct, proceed with full Stage 1 / 2 / 3
training. If anything looks wrong, fix the architecture cell and re-run the smoke test.
"""))

cells.append(code("""print("=" * 60)
print("STAGE 1 - Pre-training Individual Experts")
print("=" * 60)
set_seed(42)

expert_1 = GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2)
expert_2 = TCN_AE_Expert(D, channels=64, kernel_size=3)
expert_3 = GAT_AE_Expert(D=D, T=T_SEQ, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                          top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK)

experts      = [expert_1, expert_2, expert_3]
expert_names = ["GRU-AE", "TCN-AE", "GAT-AE"]

for name, expert in zip(expert_names, experts):
    print(f"\\n--- {name}  ({model_size_stats(expert)['Params']:,} params) ---")
    pretrain_expert(expert, Xtr_s, epochs=EXPERT_EPOCHS, lr=EXPERT_LR,
                    batch_size=BATCH_SIZE, seed=42)

# Save pre-trained weights for ablation studies later
pretrained_expert_states = [copy.deepcopy(e.state_dict()) for e in experts]
print("\\nPre-trained expert weights saved for ablation.")
"""))

cells.append(code("""print("\\n" + "=" * 60)
print("STAGE 2 - Training Gating Network (experts frozen)")
print("=" * 60)

moe = MoE_IDS(experts, input_dim=D, num_experts=N_EXPERTS,
              top_k=TOP_K, noisy_gating=True)
print(f"MoE total params: {model_size_stats(moe)['Params']:,}")

train_moe(moe, Xtr_s, epochs=GATING_EPOCHS, lr=GATING_LR,
          batch_size=BATCH_SIZE, lambda_bal=LAMBDA_BAL,
          freeze_experts=True, seed=42)

print("\\n" + "=" * 60)
print("STAGE 3 - End-to-End Fine-tuning")
print("=" * 60)

train_moe(moe, Xtr_s, epochs=E2E_EPOCHS, lr=E2E_LR,
          batch_size=BATCH_SIZE, lambda_bal=LAMBDA_BAL,
          freeze_experts=False, seed=42)

print("\\nTraining complete.")
"""))

# ------------------------------------------------------------------
# Section 7. Evaluation (verbatim)
# ------------------------------------------------------------------
cells.append(md("## 7. Evaluation\n"))

cells.append(code("""@torch.no_grad()
def moe_score_sequences(moe_model, X_seq, method="combined_mse", batch_size=512):
    moe_model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_seq, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    parts = []
    for (x,) in loader:
        parts.append(moe_model.anomaly_scores(x.to(device), method=method).cpu().numpy())
    return np.concatenate(parts)


@torch.no_grad()
def expert_score_sequences(expert, X_seq, batch_size=512):
    expert.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_seq, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    parts = []
    for (x,) in loader:
        x = x.to(device)
        parts.append(torch.mean((x - expert(x)) ** 2, dim=(1, 2)).cpu().numpy())
    return np.concatenate(parts)


@torch.no_grad()
def get_expert_routing(moe_model, X_seq, batch_size=512):
    moe_model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_seq, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    parts = []
    for (x,) in loader:
        gates, _ = moe_model.gating(x.to(device))
        parts.append(gates.cpu().numpy())
    return np.concatenate(parts)


# --- Evaluate MoE with both scoring methods ---
print("=" * 60)
print("MoE3 Option B Evaluation (Sequence Level)")
print("=" * 60)

best_method, best_roc = None, -1
for method in ["combined_mse", "weighted_expert_mse"]:
    tr_s = moe_score_sequences(moe, Xtr_s, method=method)
    te_s = moe_score_sequences(moe, Xte_s, method=method)
    m = compute_metrics_99pct(yte_seq, te_s, tr_s, pct=THR_PCT)
    print(f"\\n  {method}:")
    print(f"    ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")
    print(f"    TN={m['TN']}  FP={m['FP']}  FN={m['FN']}  TP={m['TP']}")
    if m['roc_auc'] > best_roc:
        best_roc = m['roc_auc']
        best_method = method

print(f"\\nBest scoring method: {best_method} (ROC-AUC={best_roc:.4f})")
"""))

cells.append(code("""# --- Per-attack breakdown ---
train_scores = moe_score_sequences(moe, Xtr_s, method=best_method)
test_scores  = moe_score_sequences(moe, Xte_s, method=best_method)

pa = per_attack_breakdown(test_scores, yte_seq, ate_seq)
print("Per-Attack Breakdown (MoE3 Option B):")
display(fmt_cols(pa, ["roc_auc", "pr_auc"]))

# --- Score distribution ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(test_scores[yte_seq == 0], bins=60, alpha=0.5, label="Normal", density=True)
ax.hist(test_scores[yte_seq == 1], bins=60, alpha=0.5, label="Attack", density=True)
thr = np.percentile(train_scores, THR_PCT)
ax.axvline(thr, color='r', ls='--', lw=1.5, label=f"Threshold (p{THR_PCT})")
ax.set_xlabel("Anomaly Score (MSE)"); ax.set_ylabel("Density")
ax.set_title("MoE3 Option B Score Distribution"); ax.legend()
plt.tight_layout(); plt.show()
"""))

# ------------------------------------------------------------------
# Section 8. Ablation Studies (Top-K = 1, 2, 3 — three experts)
# ------------------------------------------------------------------
cells.append(md("""## 8. Ablation Studies

1. **Individual experts** — each expert evaluated alone
2. **Simple ensembles** — mean / max / median of expert scores
3. **MoE Top-K** — sweep K = 1, 2, 3 (per Option B spec)
"""))

cells.append(code("""print("=" * 60)
print("Individual Expert Evaluation")
print("=" * 60)

expert_results = {}
for name, expert in zip(expert_names, experts):
    tr_s = expert_score_sequences(expert, Xtr_s)
    te_s = expert_score_sequences(expert, Xte_s)
    m = compute_metrics_99pct(yte_seq, te_s, tr_s, pct=THR_PCT)
    expert_results[name] = {"train_scores": tr_s, "test_scores": te_s, "metrics": m}
    print(f"  {name:20s}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}")

# --- Ensemble baselines ---
print("\\n" + "=" * 60)
print("Ensemble Baselines")
print("=" * 60)

all_tr = np.stack([expert_results[n]["train_scores"] for n in expert_names], axis=1)
all_te = np.stack([expert_results[n]["test_scores"]  for n in expert_names], axis=1)

ensemble_results = {}
for agg_name, agg_fn in [("Mean", np.mean), ("Max", np.max), ("Median", np.median)]:
    tr_ens = agg_fn(all_tr, axis=1)
    te_ens = agg_fn(all_te, axis=1)
    m = compute_metrics_99pct(yte_seq, te_ens, tr_ens, pct=THR_PCT)
    ensemble_results[agg_name] = m
    print(f"  Ensemble-{agg_name:6s}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}")
"""))

cells.append(code("""print("\\n" + "=" * 60)
print("Top-K Ablation (K = 1, 2, 3)  -- Option B sweep")
print("=" * 60)

topk_results = {}
for k in [1, 2, 3]:
    # Create fresh experts and load pre-trained weights
    exp_k = [
        GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2),
        TCN_AE_Expert(D, channels=64, kernel_size=3),
        GAT_AE_Expert(D=D, T=T_SEQ, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                      top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK),
    ]
    for dst, sd in zip(exp_k, pretrained_expert_states):
        dst.load_state_dict(sd)

    moe_k = MoE_IDS(exp_k, input_dim=D, num_experts=N_EXPERTS,
                     top_k=k, noisy_gating=True)
    print(f"\\n--- Top-{k} ---")
    train_moe(moe_k, Xtr_s, epochs=GATING_EPOCHS, lr=GATING_LR,
              batch_size=BATCH_SIZE, lambda_bal=LAMBDA_BAL,
              freeze_experts=True, seed=42)
    train_moe(moe_k, Xtr_s, epochs=E2E_EPOCHS, lr=E2E_LR,
              batch_size=BATCH_SIZE, lambda_bal=LAMBDA_BAL,
              freeze_experts=False, seed=42)

    tr_s = moe_score_sequences(moe_k, Xtr_s)
    te_s = moe_score_sequences(moe_k, Xte_s)
    m = compute_metrics_99pct(yte_seq, te_s, tr_s, pct=THR_PCT)
    topk_results[k] = m
    print(f"  Top-{k}:  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}")

print("\\nTop-K ablation complete.")
"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: Method comparison ---
labels, rocs, praucs = [], [], []
for n in expert_names:
    labels.append(n)
    rocs.append(expert_results[n]["metrics"]["roc_auc"])
    praucs.append(expert_results[n]["metrics"]["pr_auc"])
for en in ["Mean", "Max", "Median"]:
    labels.append(f"Ens-{en}")
    rocs.append(ensemble_results[en]["roc_auc"])
    praucs.append(ensemble_results[en]["pr_auc"])

moe_m = compute_metrics_99pct(yte_seq, test_scores, train_scores, pct=THR_PCT)
labels.append("MoE3 (K=2)")
rocs.append(moe_m["roc_auc"])
praucs.append(moe_m["pr_auc"])

colors = ["steelblue"]*len(expert_names) + ["forestgreen"]*3 + ["crimson"]
x = np.arange(len(labels))
axes[0].barh(x, rocs, color=colors)
for i, v in enumerate(rocs):
    axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
axes[0].set_yticks(x); axes[0].set_yticklabels(labels)
axes[0].set_xlabel("ROC-AUC"); axes[0].set_title("Method Comparison")
axes[0].set_xlim(min(rocs) - 0.05, 1.005)

# --- Right: Top-K ablation ---
ks = sorted(topk_results.keys())
axes[1].plot(ks, [topk_results[k]["roc_auc"] for k in ks], "o-", label="ROC-AUC")
axes[1].plot(ks, [topk_results[k]["pr_auc"]  for k in ks], "s--", label="PR-AUC")
axes[1].plot(ks, [topk_results[k]["f1"]      for k in ks], "^:", label="F1")
axes[1].set_xlabel("Top-K"); axes[1].set_xticks(ks)
axes[1].set_title("Top-K Ablation"); axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.show()
"""))

# ------------------------------------------------------------------
# Section 9. Routing analysis (verbatim)
# ------------------------------------------------------------------
cells.append(md("""## 9. Expert Routing Analysis

Which experts does the gating network select for different traffic patterns and
attack types?  This is the headline diagnostic for Option B — the cross-axis
diversity hypothesis predicts that GAT-AE receives meaningful weight (≥15%) on
at least one of replay (c_r) or fuzzing (f_i) attacks, in contrast to the dead
MLP-AE / LSTM-Attn-AE experts in the 4-expert version.
"""))

cells.append(code("""gates_test = get_expert_routing(moe, Xte_s)

# Average routing weights per attack type
atk_types = sorted(set(ate_seq) - {"normal"})
routing_data = {}
for atk in ["normal"] + atk_types:
    mask = ate_seq == atk
    if mask.sum() > 0:
        routing_data[atk] = gates_test[mask].mean(axis=0)

routing_df = pd.DataFrame(routing_data, index=expert_names).T
print("Average Expert Routing by Attack Type:")
display(routing_df.round(4))

# --- Acceptance-criterion check (Option B spec) ---
mean_per_expert = routing_df.mean(axis=0)
print("\\nMean routing weight across all traffic types:")
print(mean_per_expert.round(4))
crit2 = (mean_per_expert >= 0.10).all()
print(f"  [Criterion 2] all experts >= 10% mean weight: {crit2}")

gat_attacks = [a for a in atk_types if a in routing_df.index]
gat_weight_on_attacks = routing_df.loc[gat_attacks, "GAT-AE"] if gat_attacks else pd.Series(dtype=float)
crit3 = (gat_weight_on_attacks >= 0.15).any() if len(gat_weight_on_attacks) else False
print(f"  [Criterion 3] GAT-AE >= 15% on at least one attack:  {crit3}")
if len(gat_weight_on_attacks):
    print(f"    GAT-AE weights per attack type: {gat_weight_on_attacks.round(4).to_dict()}")

# Heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.heatmap(routing_df, annot=True, fmt=".3f", cmap="YlOrRd",
            ax=axes[0], vmin=0, vmax=1)
axes[0].set_title("Expert Routing Weights by Attack Type")
axes[0].set_ylabel("Traffic Type"); axes[0].set_xlabel("Expert")

# Primary expert per attack
primary = routing_df.idxmax(axis=1)
print("\\nPrimary expert per attack type:")
for atk, exp in primary.items():
    print(f"  {atk:15s} -> {exp}")

# Stacked bar chart
routing_df.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2")
axes[1].set_title("Expert Allocation per Attack Type")
axes[1].set_ylabel("Gate Weight"); axes[1].set_xlabel("")
axes[1].legend(title="Expert", bbox_to_anchor=(1.05, 1))
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout(); plt.show()
"""))

# ------------------------------------------------------------------
# Section 10. Confusion Matrices (verbatim)
# ------------------------------------------------------------------
cells.append(md("## 10. Confusion Matrices\n"))

cells.append(code("""thr = np.percentile(train_scores, THR_PCT)
y_pred = (test_scores > thr).astype(int)

n_attacks = len(atk_types)
fig, axes = plt.subplots(1, min(n_attacks + 1, 5), figsize=(5 * min(n_attacks + 1, 5), 4.5))
if not isinstance(axes, np.ndarray):
    axes = [axes]

# Overall
cm = confusion_matrix(yte_seq, y_pred, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt="d", ax=axes[0],
            xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"],
            cmap="Blues")
axes[0].set_title("Overall"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

# Per-attack (up to 4)
for i, atk in enumerate(atk_types[:min(n_attacks, 4)]):
    mask = np.isin(ate_seq, ["normal", atk])
    y_sub = yte_seq[mask]; p_sub = y_pred[mask]
    cm_a = confusion_matrix(y_sub, p_sub, labels=[0, 1])
    ax = axes[i + 1] if i + 1 < len(axes) else axes[-1]
    sns.heatmap(cm_a, annot=True, fmt="d", ax=ax,
                xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"],
                cmap="Oranges")
    ax.set_title(atk); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

plt.suptitle("MoE3 Option B Confusion Matrices (Sequence Level)", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()

# Print summary
print(f"Overall:  Precision={precision_score(yte_seq, y_pred):.4f}  "
      f"Recall={recall_score(yte_seq, y_pred):.4f}  "
      f"F1={f1_score(yte_seq, y_pred):.4f}")
"""))

# ------------------------------------------------------------------
# Section 11. Latency / Energy / Param sizes (verbatim adapted to 3 experts)
# ------------------------------------------------------------------
cells.append(md("## 11. Model Size, Latency & Energy\n"))

cells.append(code("""print("MoE3 Option B Component Sizes:")
for name, expert in zip(expert_names, moe.experts):
    s = model_size_stats(expert)
    print(f"  {name:20s}  {s['Params']:>8,} params  {s['Size (MB)']:.4f} MB")
s_gate = model_size_stats(moe.gating)
print(f"  {'Gating Network':20s}  {s_gate['Params']:>8,} params  {s_gate['Size (MB)']:.4f} MB")
s_total = model_size_stats(moe)
print(f"  {'TOTAL MoE3':20s}  {s_total['Params']:>8,} params  {s_total['Size (MB)']:.4f} MB")

# --- Acceptance-criterion 5 check (Option B spec): total params < 250K ---
crit5 = s_total["Params"] < 250_000
print(f"  [Criterion 5] total params < 250K: {crit5}  ({s_total['Params']:,})")

# --- Inference latency ---
x_sample = torch.tensor(Xte_s[:1], dtype=torch.float32, device=device)

def moe_one_inference():
    with torch.no_grad():
        moe.anomaly_scores(x_sample, method=best_method)

moe_latency_ms = measure_latency_ms(moe_one_inference, runs=500)
print(f"\\nMoE3 inference latency: {moe_latency_ms:.4f} ms/seq")

# --- Acceptance-criterion 6 check: latency < 3 ms/seq ---
crit6 = moe_latency_ms < 3.0
print(f"  [Criterion 6] latency < 3 ms/seq: {crit6}")

# --- Individual expert latencies ---
for name, expert in zip(expert_names, moe.experts):
    expert.eval()
    def _inf(e=expert):
        with torch.no_grad(): e(x_sample)
    lat = measure_latency_ms(_inf, runs=500)
    print(f"  {name:20s}  {lat:.4f} ms/seq")

# --- Energy ---
try:
    E, P, T = nvml_energy_joules(repeat_workload(moe_one_inference, n=500))
    moe_energy_j = E / 500
    print(f"\\nMoE3 energy: {moe_energy_j:.6f} J/seq,  avg power: {P:.2f} W")
except Exception as exc:
    print(f"Energy measurement failed: {exc}")
    moe_energy_j = np.nan
"""))

# ------------------------------------------------------------------
# Section 12. Final comparison table (with MoE-IDS 4-expert + INDRA/LATTE/TENET)
# ------------------------------------------------------------------
cells.append(md("""## 12. Final Comparison with Baselines

Sequence-level results vs the 4-expert MoE-IDS and the INDRA / LATTE / TENET / GAN
baselines from `TOW-IDS-ML.ipynb` and `TOW-IDS-MoE.ipynb`.
"""))

cells.append(code("""# Baseline results from TOW-IDS-ML / TOW-IDS-MoE
baselines = [
    {"Model": "INDRA (GRU-AE)",   "ROC-AUC": 0.990, "PR-AUC": 0.980,
     "Params": 64_652,  "Latency (ms)": 1.29,  "Energy (J)": 0.015},
    {"Model": "MAD-GAN",          "ROC-AUC": 0.983, "PR-AUC": 0.990,
     "Params": 41_805,  "Latency (ms)": 40.21, "Energy (J)": 0.847},
    {"Model": "LATTE (thr)",      "ROC-AUC": 0.977, "PR-AUC": 0.944,
     "Params": 164_236, "Latency (ms)": 0.20,  "Energy (J)": 0.024},
    {"Model": "TENET+DT",         "ROC-AUC": 0.975, "PR-AUC": 0.946,
     "Params": 22_728,  "Latency (ms)": 3.17,  "Energy (J)": 0.074},
    {"Model": "TadGAN",           "ROC-AUC": 0.913, "PR-AUC": 0.954,
     "Params": 62_813,  "Latency (ms)": 0.72,  "Energy (J)": 0.023},
    {"Model": "GANomaly",         "ROC-AUC": 0.706, "PR-AUC": 0.739,
     "Params": 1_553,   "Latency (ms)": 0.25,  "Energy (J)": 0.009},
    # MoE-IDS 4-expert (from TOW-IDS-MoE.ipynb) — fill in once available
    # {"Model": "MoE-IDS 4-expert", ...},
]

# Add MoE3 result
moe_final = compute_metrics_99pct(yte_seq, test_scores, train_scores, pct=THR_PCT)
baselines.append({
    "Model":        "MoE3 Option B (Ours)",
    "ROC-AUC":      round(moe_final["roc_auc"], 4),
    "PR-AUC":       round(moe_final["pr_auc"], 4),
    "Params":       model_size_stats(moe)["Params"],
    "Latency (ms)": round(moe_latency_ms, 4),
    "Energy (J)":   round(moe_energy_j, 6) if not np.isnan(moe_energy_j) else "N/A",
})

comp_df = pd.DataFrame(baselines).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print("Final Deployment Comparison:")
display(comp_df)

# --- Bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = comp_df["Model"].tolist()
colors = ["crimson" if "MoE3" in m else "steelblue" for m in models]

axes[0].barh(models, comp_df["ROC-AUC"], color=colors)
for i, v in enumerate(comp_df["ROC-AUC"]):
    axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
axes[0].set_xlabel("ROC-AUC"); axes[0].set_title("ROC-AUC Comparison")
axes[0].set_xlim(0.6, 1.01)

axes[1].barh(models, comp_df["PR-AUC"], color=colors)
for i, v in enumerate(comp_df["PR-AUC"]):
    axes[1].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
axes[1].set_xlabel("PR-AUC"); axes[1].set_title("PR-AUC Comparison")
axes[1].set_xlim(0.6, 1.01)

plt.suptitle("MoE3 Option B vs Baseline Models", fontsize=14)
plt.tight_layout(); plt.show()

# --- Save results ---
os.makedirs("./results", exist_ok=True)
comp_df.to_csv("./results/MoE3_OptionB_results.csv", index=False)
print("\\nResults written to ./results/MoE3_OptionB_results.csv")
print("Notebook complete.")
"""))

# ------------------------------------------------------------------
# Notebook metadata
# ------------------------------------------------------------------
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

# Each cell needs an id (nbformat 4.5+ requires it)
import uuid
for c in cells:
    c["id"] = uuid.uuid4().hex[:12]

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Wrote: {OUT_PATH}")
print(f"Total cells: {len(cells)}")
