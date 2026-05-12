"""Convergence diagnostic for MoE3-IDS Option B (seed=42).

Re-runs Stage 1 with full per-epoch loss capture (deterministic), then evaluates
each expert standalone (equivalent to forcing K=1 hard routing to that expert).
Saves results/MoE3_OptionB_diagnostic.csv.
"""
import os, random, copy, json
import numpy as np
import pandas as pd
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

WORKDIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKDIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Config
# ============================================================
OUT_DIR      = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/extracted"
y_train_path = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_train.csv"
y_test_path  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_test.csv"

WINDOW_SEC = 1.0
T_SEQ      = 40
THR_PCT    = 99
SEED       = 42
EXPERT_EPOCHS = 100
EXPERT_LR     = 1e-3
BATCH_SIZE    = 256

GAT_HIDDEN     = 32
GAT_HEADS      = 2
GAT_TOPK       = 4
GAT_BOTTLENECK = 16


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
Xtr_seq, _, _ = make_sequences(
    train_w[train_w["y_window"] == 0].reset_index(drop=True),
    t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
Xte_seq, yte_seq, ate_seq = make_sequences(
    test_w, t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
scaler = StandardScaler().fit(Xtr_seq.reshape(-1, D))
Xtr_s = scaler.transform(Xtr_seq.reshape(-1, D)).reshape(Xtr_seq.shape).astype(np.float32)
Xte_s = scaler.transform(Xte_seq.reshape(-1, D)).reshape(Xte_seq.shape).astype(np.float32)
print(f"[data] D={D}, Xtr_s={Xtr_s.shape}, Xte_s={Xte_s.shape}")


# ============================================================
# Expert architectures
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


# ============================================================
# Pretrain with full per-epoch logging
# ============================================================
def pretrain_with_curve(expert, Xtr_s, *, epochs=100, lr=1e-3, batch_size=256, seed=0):
    set_seed(seed)
    model = expert.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    curve = []
    for ep in range(epochs):
        losses = []
        for (x,) in loader:
            x = x.to(device)
            loss = loss_fn(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        curve.append(float(np.mean(losses)))
    return model, curve


# ============================================================
# Standalone evaluation (equivalent to K=1 hard routing to expert)
# ============================================================
@torch.no_grad()
def expert_score_sequences(expert, X_seq, batch_size=512):
    """Per-sequence MSE under expert.

    Mathematically identical to evaluating an MoE_IDS wrapper that has hard-routed
    100% of its gate weight to this single expert (K=1 to this expert in turn),
    because gate_e = 1.0 collapses combined_mse to mean((x - expert_e(x))^2).
    """
    expert.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_seq, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    parts = []
    for (x,) in loader:
        x = x.to(device)
        parts.append(torch.mean((x - expert(x)) ** 2, dim=(1, 2)).cpu().numpy())
    return np.concatenate(parts)


def metrics_at_p99(y_true, scores, train_scores, pct=99):
    thr = float(np.percentile(np.asarray(train_scores, dtype=float), pct))
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc":   float(roc_auc_score(y_true, scores)),
        "pr_auc":    float(average_precision_score(y_true, scores)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "thr":       thr,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


# ============================================================
# Run diagnostic
# ============================================================
print("\n" + "=" * 60)
print(f"DIAGNOSTIC - Stage 1 convergence + standalone eval (seed={SEED})")
print("=" * 60)

set_seed(SEED)
expert_specs = [
    ("GRU-AE", lambda: GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2)),
    ("TCN-AE", lambda: TCN_AE_Expert(D, channels=64, kernel_size=3)),
    ("GAT-AE", lambda: GAT_AE_Expert(D=D, T=T_SEQ, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                                       top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK)),
]

curves = {}
trained = {}
for name, ctor in expert_specs:
    print(f"\n--- {name} ---")
    expert = ctor()
    n_params = sum(p.numel() for p in expert.parameters())
    print(f"  params: {n_params:,}")
    expert, curve = pretrain_with_curve(expert, Xtr_s, epochs=EXPERT_EPOCHS,
                                          lr=EXPERT_LR, batch_size=BATCH_SIZE, seed=SEED)
    curves[name]  = curve
    trained[name] = expert
    # Print sparse-epoch checkpoints
    print(f"  loss @ ep 1  : {curve[0]:.6f}")
    print(f"  loss @ ep 25 : {curve[24]:.6f}")
    print(f"  loss @ ep 50 : {curve[49]:.6f}")
    print(f"  loss @ ep 75 : {curve[74]:.6f}")
    print(f"  loss @ ep 100: {curve[99]:.6f}")
    print(f"  final loss   : {curve[-1]:.6f}")


# ============================================================
# Standalone evaluation (force K=1 hard routing to each expert)
# ============================================================
print("\n" + "=" * 60)
print("STANDALONE EVALUATION (K=1 forced to each expert)")
print("=" * 60)

standalone = {}
for name, expert in trained.items():
    tr_s = expert_score_sequences(expert, Xtr_s)
    te_s = expert_score_sequences(expert, Xte_s)
    m = metrics_at_p99(yte_seq, te_s, tr_s, pct=THR_PCT)
    standalone[name] = m
    print(f"\n  {name}")
    print(f"    ROC-AUC = {m['roc_auc']:.4f}")
    print(f"    PR-AUC  = {m['pr_auc']:.4f}")
    print(f"    F1      = {m['f1']:.4f}")
    print(f"    P/R     = {m['precision']:.4f} / {m['recall']:.4f}")
    print(f"    TN={m['TN']}  FP={m['FP']}  FN={m['FN']}  TP={m['TP']}")


# ============================================================
# Build diagnostic table and save CSV
# ============================================================
rows = []
for name in [n for n, _ in expert_specs]:
    c = curves[name]
    m = standalone[name]
    n_params = sum(p.numel() for p in trained[name].parameters())
    rows.append({
        "expert":         name,
        "params":         n_params,
        "loss_ep1":       c[0],
        "loss_ep25":      c[24],
        "loss_ep50":      c[49],
        "loss_ep75":      c[74],
        "loss_ep100":     c[99],
        "loss_final":     c[-1],
        "loss_min":       min(c),
        "loss_min_epoch": int(np.argmin(c)) + 1,
        "standalone_roc_auc": m["roc_auc"],
        "standalone_pr_auc":  m["pr_auc"],
        "standalone_f1":      m["f1"],
        "standalone_precision": m["precision"],
        "standalone_recall":    m["recall"],
        "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
    })
diag_df = pd.DataFrame(rows)
diag_csv = os.path.join(RESULTS_DIR, "MoE3_OptionB_diagnostic.csv")
diag_df.to_csv(diag_csv, index=False)

# Save the full per-epoch loss curves as a side artifact (one column per expert)
curve_df = pd.DataFrame({"epoch": list(range(1, EXPERT_EPOCHS + 1)), **curves})
curve_csv = os.path.join(RESULTS_DIR, "MoE3_OptionB_loss_curves.csv")
curve_df.to_csv(curve_csv, index=False)


# ============================================================
# Print final table
# ============================================================
print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)
display_cols = ["expert", "params", "loss_ep1", "loss_ep25", "loss_ep50", "loss_ep75",
                "loss_ep100", "loss_final", "standalone_roc_auc", "standalone_pr_auc",
                "standalone_f1"]
fmt = {c: "{:.6f}".format if "loss" in c else
          "{:.4f}".format if "standalone" in c else str
       for c in display_cols}
print(diag_df[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print(f"\nSaved: {diag_csv}")
print(f"Saved: {curve_csv}")
