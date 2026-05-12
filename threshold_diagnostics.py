"""D1 + D2 diagnostic for MoE3-IDS Option B.

D1 — Threshold sensitivity sweep. For each of GRU-AE, TCN-AE, GAT-AE, MoE,
     compute precision/recall/F1 at percentile thresholds {90, 92, 95, 97, 99,
     99.5, 99.9}. Plot PR curves with threshold markers. Saves CSV + PNG.

D2 — Validation-set thresholding. Carve a 50-sequence held-out validation
     set from the 234-sequence train pool. Retrain Stage 1 / 2 / 3 on the
     remaining 184. Compare the original train-p99 threshold to a
     validation-p99 threshold. Saves CSV.

Both diagnostics use seed=42 for reproducibility.
"""
import os, random, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, precision_recall_curve,
)

WORKDIR        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(WORKDIR, "checkpoints")
RESULTS_DIR    = os.path.join(WORKDIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Config (mirrors full_train_moe3.py)
# ============================================================
OUT_DIR      = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/extracted"
y_train_path = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_train.csv"
y_test_path  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_test.csv"

WINDOW_SEC = 1.0
T_SEQ      = 40
SEED       = 42

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

GAT_HIDDEN     = 32
GAT_HEADS      = 2
GAT_TOPK       = 4
GAT_BOTTLENECK = 16

VAL_N = 50  # hold-out validation count for D2

PERCENTILES_D1 = [90, 92, 95, 97, 99, 99.5, 99.9]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_feature_cols(df):
    return [c for c in df.columns if c not in ["w", "y_window", "attack_type_window"]]


# ============================================================
# Data ingestion (verbatim)
# ============================================================
print("\n[data] Loading cached packet CSVs ...")
train_csv = f"{OUT_DIR}/packets_train.csv"
test_csv  = f"{OUT_DIR}/packets_test.csv"
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


def build_global_windows_with_attack(df, window_sec=1.0):
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


print("[data] Building train/test windows ...")
train_w = build_global_windows_with_attack(train_pkts, window_sec=WINDOW_SEC)
test_w  = build_global_windows_with_attack(test_pkts,  window_sec=WINDOW_SEC)
feat_cols = get_feature_cols(train_w)
D = len(feat_cols)
Xtr_seq_full, _, _ = make_sequences(
    train_w[train_w["y_window"] == 0].reset_index(drop=True),
    t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
Xte_seq, yte_seq, ate_seq = make_sequences(
    test_w, t_seq=T_SEQ, stride=1, feature_cols=feat_cols)

# Two scaler scenarios:
#   (D1) Use the original full-pool scaler/data — same as full_train_moe3.py
#   (D2) Carve a 50-sequence validation set, refit scaler on the 184-train,
#        retrain experts and MoE on 184-train, evaluate.
# We compute D1's scaler+data here; D2 reproduces below.
scaler_full = StandardScaler().fit(Xtr_seq_full.reshape(-1, D))
Xtr_full = scaler_full.transform(Xtr_seq_full.reshape(-1, D)).reshape(Xtr_seq_full.shape).astype(np.float32)
Xte_full = scaler_full.transform(Xte_seq.reshape(-1, D)).reshape(Xte_seq.shape).astype(np.float32)
print(f"[data] D={D}, Xtr_full={Xtr_full.shape}, Xte_full={Xte_full.shape}")


# ============================================================
# Architectures (verbatim)
# ============================================================
class GRU_AE_Expert(nn.Module):
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


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, heads * out_dim)
        self.proj = nn.Linear(heads * out_dim, out_dim)
    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.out_dim)
        outs = [torch.bmm(adj, h[:, :, i, :]) for i in range(self.heads)]
        return self.proj(torch.cat(outs, dim=-1))


class GAT_AE_Expert(nn.Module):
    def __init__(self, D=12, T=40, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                 top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK):
        super().__init__()
        self.D, self.T = D, T
        self.top_k = min(top_k, D)
        self.enc_lin1  = nn.Linear(T, hidden)
        self.enc_attn1 = MultiHeadGraphAttention(hidden, hidden,    heads=heads)
        self.enc_attn2 = MultiHeadGraphAttention(hidden, bottleneck, heads=heads)
        self.dec_attn1 = MultiHeadGraphAttention(bottleneck, hidden, heads=heads)
        self.dec_attn2 = MultiHeadGraphAttention(hidden,     hidden, heads=heads)
        self.dec_lin   = nn.Linear(hidden, T)
    def build_dynamic_adj(self, x_feat):
        x_norm = F.normalize(x_feat, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
        topk_vals, topk_idx = sim.topk(self.top_k, dim=-1)
        adj = torch.zeros_like(sim)
        adj.scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))
        return adj
    def forward(self, x):
        x_feat = x.transpose(1, 2)
        adj = self.build_dynamic_adj(x_feat)
        h = F.elu(self.enc_lin1(x_feat))
        h = F.elu(self.enc_attn1(h, adj))
        z = self.enc_attn2(h, adj)
        h = F.elu(self.dec_attn1(z, adj))
        h = F.elu(self.dec_attn2(h, adj))
        return self.dec_lin(h).transpose(1, 2)


class SparseGatingNetwork(nn.Module):
    def __init__(self, seq_dim, num_experts, top_k=2, noisy=True):
        super().__init__()
        input_dim = seq_dim * 2
        self.num_experts = num_experts; self.top_k = top_k; self.noisy = noisy
        self.gate = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, num_experts))
        if noisy:
            self.noise_linear = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, num_experts))
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


class MoE_IDS(nn.Module):
    def __init__(self, experts, input_dim, num_experts=4, top_k=2, noisy_gating=True):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating = SparseGatingNetwork(input_dim, num_experts, top_k, noisy_gating)
        self.num_experts = num_experts; self.top_k = top_k
    def forward(self, x):
        gates, raw_logits = self.gating(x)
        recons = torch.stack([e(x) for e in self.experts], dim=1)
        g = gates.unsqueeze(-1).unsqueeze(-1)
        return (g * recons).sum(dim=1), gates, raw_logits, recons
    def anomaly_scores(self, x, method='combined_mse'):
        output, gates, _, recons = self.forward(x)
        if method == 'combined_mse':
            return torch.mean((x - output) ** 2, dim=(1, 2))
        elif method == 'weighted_expert_mse':
            emse = torch.mean((x.unsqueeze(1) - recons) ** 2, dim=(2, 3))
            return (gates * emse).sum(dim=1)


def load_balancing_loss(gates, raw_logits, num_experts):
    f = gates.mean(dim=0)
    P = F.softmax(raw_logits, dim=-1).mean(dim=0)
    return num_experts * (f * P).sum()


# ============================================================
# Score helpers
# ============================================================
@torch.no_grad()
def expert_scores(expert, X, batch=512):
    expert.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch, shuffle=False)
    parts = []
    for (x,) in loader:
        x = x.to(device)
        parts.append(torch.mean((x - expert(x)) ** 2, dim=(1, 2)).cpu().numpy())
    return np.concatenate(parts)


@torch.no_grad()
def moe_scores(moe, X, method="weighted_expert_mse", batch=512):
    moe.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch, shuffle=False)
    parts = []
    for (x,) in loader:
        parts.append(moe.anomaly_scores(x.to(device), method=method).cpu().numpy())
    return np.concatenate(parts)


def metrics_at_thr(y_true, scores, thr):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "thr":       float(thr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


# ============================================================
# D1 — load checkpoint, sweep thresholds
# ============================================================
print("\n" + "=" * 60)
print("D1 — Threshold sensitivity sweep")
print("=" * 60)

ckpt_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionB_seed{SEED}.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
print(f"  Loaded checkpoint: {ckpt_path}")

# Reconstruct experts from pretrained_experts state dicts
gru_d1 = GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2).to(device)
tcn_d1 = TCN_AE_Expert(D, channels=64, kernel_size=3).to(device)
gat_d1 = GAT_AE_Expert(D=D, T=T_SEQ, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                        top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK).to(device)
gru_d1.load_state_dict(ckpt["pretrained_experts"][0])
tcn_d1.load_state_dict(ckpt["pretrained_experts"][1])
gat_d1.load_state_dict(ckpt["pretrained_experts"][2])

# Reconstruct MoE — note: MoE in checkpoint was end-to-end fine-tuned (Stages 2+3),
# so we load the MoE state dict (which contains both experts and gating)
moe_d1 = MoE_IDS([
    GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2),
    TCN_AE_Expert(D, channels=64, kernel_size=3),
    GAT_AE_Expert(D=D, T=T_SEQ, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                  top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK),
], input_dim=D, num_experts=N_EXPERTS, top_k=TOP_K, noisy_gating=True).to(device)
moe_d1.load_state_dict(ckpt["moe_state_dict"])

# Compute scores once on full train and test pools
models_d1 = {
    "GRU-AE": (lambda X: expert_scores(gru_d1, X)),
    "TCN-AE": (lambda X: expert_scores(tcn_d1, X)),
    "GAT-AE": (lambda X: expert_scores(gat_d1, X)),
    "MoE":    (lambda X: moe_scores(moe_d1, X, method="weighted_expert_mse")),
}

train_scores_d1 = {n: f(Xtr_full) for n, f in models_d1.items()}
test_scores_d1  = {n: f(Xte_full) for n, f in models_d1.items()}

# Sweep
rows_d1 = []
for name in models_d1:
    tr_s = train_scores_d1[name]
    te_s = test_scores_d1[name]
    roc = roc_auc_score(yte_seq, te_s)
    pra = average_precision_score(yte_seq, te_s)
    for pct in PERCENTILES_D1:
        thr = float(np.percentile(tr_s, pct))
        m = metrics_at_thr(yte_seq, te_s, thr)
        rows_d1.append({
            "model":     name,
            "percentile": pct,
            "thr":       thr,
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
            "roc_auc":   roc,
            "pr_auc":    pra,
        })
df_d1 = pd.DataFrame(rows_d1)
d1_csv = os.path.join(RESULTS_DIR, "MoE3_OptionB_threshold_sweep.csv")
df_d1.to_csv(d1_csv, index=False)
print(f"\n  Saved: {d1_csv}")

# Pretty-print as a pivot
print("\n  D1 — F1 at percentile threshold (from train-normal scores):")
piv_f1 = df_d1.pivot(index="model", columns="percentile", values="f1")
print(piv_f1.round(4).to_string())
print("\n  D1 — Precision at percentile threshold:")
piv_p = df_d1.pivot(index="model", columns="percentile", values="precision")
print(piv_p.round(4).to_string())
print("\n  D1 — Recall at percentile threshold:")
piv_r = df_d1.pivot(index="model", columns="percentile", values="recall")
print(piv_r.round(4).to_string())

# PR curve plot with threshold markers
fig, ax = plt.subplots(figsize=(8.5, 6))
colors = {"GRU-AE": "steelblue", "TCN-AE": "darkorange", "GAT-AE": "forestgreen", "MoE": "crimson"}
for name in models_d1:
    te_s = test_scores_d1[name]
    p_curve, r_curve, _ = precision_recall_curve(yte_seq, te_s)
    ax.plot(r_curve, p_curve, label=f"{name}", color=colors[name], lw=2)
    # Threshold markers
    for pct in PERCENTILES_D1:
        sub = df_d1[(df_d1["model"] == name) & (df_d1["percentile"] == pct)].iloc[0]
        ax.scatter(sub["recall"], sub["precision"], color=colors[name], s=40,
                   edgecolors="black", linewidths=0.6, zorder=5)
        if pct in (90, 99, 99.9):
            ax.annotate(f"p{pct}", (sub["recall"], sub["precision"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=colors[name])
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("D1 — Precision-Recall curves with train-percentile threshold markers")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.78, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left")
plt.tight_layout()
d1_png = os.path.join(RESULTS_DIR, "MoE3_OptionB_threshold_sweep.png")
plt.savefig(d1_png, dpi=130)
plt.close()
print(f"  Saved: {d1_png}")


# ============================================================
# D2 — validation-set thresholding (retrain on 184, threshold from 50)
# ============================================================
print("\n" + "=" * 60)
print(f"D2 — Validation-set thresholding (carve {VAL_N} of 234, retrain on 184)")
print("=" * 60)

# Deterministic 184/50 split. Use np.random.default_rng for reproducibility.
rng = np.random.default_rng(SEED)
N_train_full = Xtr_seq_full.shape[0]
idx = np.arange(N_train_full)
rng.shuffle(idx)
val_idx = np.sort(idx[:VAL_N])
trn_idx = np.sort(idx[VAL_N:])
print(f"  split: {len(trn_idx)} train, {len(val_idx)} validation, seed={SEED}")

# Refit scaler on the new train pool only (no leakage)
Xtr_raw  = Xtr_seq_full[trn_idx]
Xval_raw = Xtr_seq_full[val_idx]
scaler_d2 = StandardScaler().fit(Xtr_raw.reshape(-1, D))
def sx(arr):
    n, t, d = arr.shape
    return scaler_d2.transform(arr.reshape(-1, d)).reshape(n, t, d).astype(np.float32)
Xtr_d2  = sx(Xtr_raw)
Xval_d2 = sx(Xval_raw)
Xte_d2  = sx(Xte_seq)
print(f"  Xtr_d2={Xtr_d2.shape}, Xval_d2={Xval_d2.shape}, Xte_d2={Xte_d2.shape}")

# --- Retrain Stage 1 on 184 ---
def pretrain(expert, X, *, epochs=100, lr=1e-3, batch_size=256, seed=0):
    set_seed(seed)
    model = expert.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            loss = loss_fn(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_moe(moe_model, X, *, epochs=50, lr=1e-4, batch_size=256,
              lambda_bal=0.01, freeze_experts=True, seed=0):
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
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(params, lr=lr); loss_fn = nn.MSELoss()
    moe_model.train()
    for ep in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            output, gates, raw_logits, _ = moe_model(x)
            rec = loss_fn(output, x)
            bal = load_balancing_loss(gates, raw_logits, moe_model.num_experts)
            loss = rec + lambda_bal * bal
            opt.zero_grad(); loss.backward(); opt.step()
    return moe_model


print("\n  Stage 1 — pre-training experts on 184 sequences")
set_seed(SEED)
gru_d2 = pretrain(GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2),
                  Xtr_d2, epochs=EXPERT_EPOCHS, lr=EXPERT_LR, batch_size=BATCH_SIZE, seed=SEED)
print(f"    GRU-AE done")
tcn_d2 = pretrain(TCN_AE_Expert(D, channels=64, kernel_size=3),
                  Xtr_d2, epochs=EXPERT_EPOCHS, lr=EXPERT_LR, batch_size=BATCH_SIZE, seed=SEED)
print(f"    TCN-AE done")
gat_d2 = pretrain(GAT_AE_Expert(D=D, T=T_SEQ, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                                  top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK),
                  Xtr_d2, epochs=EXPERT_EPOCHS, lr=EXPERT_LR, batch_size=BATCH_SIZE, seed=SEED)
print(f"    GAT-AE done")

print("\n  Stages 2 + 3 — gating + e2e on 184 sequences")
moe_d2 = MoE_IDS([gru_d2, tcn_d2, gat_d2], input_dim=D,
                  num_experts=N_EXPERTS, top_k=TOP_K, noisy_gating=True).to(device)
train_moe(moe_d2, Xtr_d2, epochs=GATING_EPOCHS, lr=GATING_LR,
          batch_size=BATCH_SIZE, lambda_bal=LAMBDA_BAL, freeze_experts=True, seed=SEED)
train_moe(moe_d2, Xtr_d2, epochs=E2E_EPOCHS, lr=E2E_LR,
          batch_size=BATCH_SIZE, lambda_bal=LAMBDA_BAL, freeze_experts=False, seed=SEED)
print("    Stage 2+3 done")

# --- Evaluate at both thresholds ---
models_d2 = {
    "GRU-AE": (lambda X: expert_scores(gru_d2, X)),
    "TCN-AE": (lambda X: expert_scores(tcn_d2, X)),
    "GAT-AE": (lambda X: expert_scores(gat_d2, X)),
    "MoE":    (lambda X: moe_scores(moe_d2, X, method="weighted_expert_mse")),
}

rows_d2 = []
for name, fn in models_d2.items():
    tr_s  = fn(Xtr_d2)
    val_s = fn(Xval_d2)
    te_s  = fn(Xte_d2)
    roc = roc_auc_score(yte_seq, te_s)
    pra = average_precision_score(yte_seq, te_s)

    thr_train = float(np.percentile(tr_s, 99))
    thr_val   = float(np.percentile(val_s, 99))
    m_train = metrics_at_thr(yte_seq, te_s, thr_train)
    m_val   = metrics_at_thr(yte_seq, te_s, thr_val)
    rows_d2.append({
        "model":         name,
        "thr_method":    "train_p99",
        "thr":           thr_train,
        "precision":     m_train["precision"],
        "recall":        m_train["recall"],
        "f1":            m_train["f1"],
        "TN": m_train["TN"], "FP": m_train["FP"], "FN": m_train["FN"], "TP": m_train["TP"],
        "roc_auc":       roc,
        "pr_auc":        pra,
        "train_score_mean": float(np.mean(tr_s)),
        "val_score_mean":   float(np.mean(val_s)),
        "test_normal_score_mean": float(np.mean(te_s[yte_seq == 0])),
    })
    rows_d2.append({
        "model":         name,
        "thr_method":    "val_p99",
        "thr":           thr_val,
        "precision":     m_val["precision"],
        "recall":        m_val["recall"],
        "f1":            m_val["f1"],
        "TN": m_val["TN"], "FP": m_val["FP"], "FN": m_val["FN"], "TP": m_val["TP"],
        "roc_auc":       roc,
        "pr_auc":        pra,
        "train_score_mean": float(np.mean(tr_s)),
        "val_score_mean":   float(np.mean(val_s)),
        "test_normal_score_mean": float(np.mean(te_s[yte_seq == 0])),
    })

df_d2 = pd.DataFrame(rows_d2)
d2_csv = os.path.join(RESULTS_DIR, "MoE3_OptionB_validation_threshold.csv")
df_d2.to_csv(d2_csv, index=False)
print(f"\n  Saved: {d2_csv}")

# Pretty print
print("\n  D2 — Train-p99 vs Validation-p99 thresholds (test set evaluation):")
piv_thr = df_d2.pivot(index="model", columns="thr_method", values="thr").round(4)
piv_pre = df_d2.pivot(index="model", columns="thr_method", values="precision").round(4)
piv_rec = df_d2.pivot(index="model", columns="thr_method", values="recall").round(4)
piv_f1  = df_d2.pivot(index="model", columns="thr_method", values="f1").round(4)
combined = pd.concat({
    "thr":       piv_thr,
    "precision": piv_pre,
    "recall":    piv_rec,
    "f1":        piv_f1,
}, axis=1)
print(combined.to_string())

print("\n  Score-distribution sanity (mean of model scores on train-184 / val-50 / test-normal-62):")
sane = df_d2[df_d2["thr_method"] == "train_p99"][[
    "model", "train_score_mean", "val_score_mean", "test_normal_score_mean"
]].round(6)
print(sane.to_string(index=False))

print("\n" + "=" * 60)
print("D1 + D2 complete. CSV artifacts in ./results/")
print("=" * 60)
