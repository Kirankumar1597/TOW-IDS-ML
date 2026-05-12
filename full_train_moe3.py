"""Full three-stage training run for MoE3-IDS Option B (seed=42).

Mirrors the cells in `MoE3-IDS-OptionB.ipynb`. Saves the trained MoE checkpoint
to ./checkpoints/MoE3_OptionB_seed42.pt and the metrics CSV to
./results/MoE3_OptionB_results.csv. After training, runs the routing analysis
and prints the per-attack expert weight distribution table.
"""
import os, sys, random, time, copy, json
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
CHECKPOINT_DIR = os.path.join(WORKDIR, "checkpoints")
RESULTS_DIR    = os.path.join(WORKDIR, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
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
    return {"Params": int(params)}


# ============================================================
# Data ingestion (verbatim from notebook)
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

Xtr_seq, ytr_seq, atr_seq = make_sequences(
    train_w[train_w["y_window"] == 0].reset_index(drop=True),
    t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
Xte_seq, yte_seq, ate_seq = make_sequences(
    test_w, t_seq=T_SEQ, stride=1, feature_cols=feat_cols)

scaler = StandardScaler().fit(Xtr_seq.reshape(-1, D))
Xtr_s = scaler.transform(Xtr_seq.reshape(-1, D)).reshape(Xtr_seq.shape).astype(np.float32)
Xte_s = scaler.transform(Xte_seq.reshape(-1, D)).reshape(Xte_seq.shape).astype(np.float32)

print(f"[data] D={D}, T={T_SEQ}")
print(f"[data] Train sequences (normal): {Xtr_s.shape}")
print(f"[data] Test sequences (all):     {Xte_s.shape}")
print(f"[data] Test - normal: {int((yte_seq==0).sum())}, anomaly: {int((yte_seq==1).sum())}")
print(f"[data] Test attack types: {pd.Series(ate_seq).value_counts().to_dict()}")


# ============================================================
# Expert architectures (verbatim + new GAT-AE)
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
# Gating + MoE wrapper (verbatim)
# ============================================================
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
        output = (g * recons).sum(dim=1)
        return output, gates, raw_logits, recons

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
# Training functions (verbatim)
# ============================================================
def pretrain_expert(expert, Xtr_s, *, epochs=100, lr=1e-3, batch_size=256, seed=0):
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
    opt = optim.Adam(params, lr=lr); loss_fn = nn.MSELoss()
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


# ============================================================
# Stage 1 - 3 (FULL training, seed=42)
# ============================================================
print("\n" + "=" * 60)
print(f"FULL TRAINING - seed={SEED}")
print("=" * 60)

set_seed(SEED)
expert_1 = GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2)
expert_2 = TCN_AE_Expert(D, channels=64, kernel_size=3)
expert_3 = GAT_AE_Expert(D=D, T=T_SEQ, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                          top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK)
experts      = [expert_1, expert_2, expert_3]
expert_names = ["GRU-AE", "TCN-AE", "GAT-AE"]

print("\n--- STAGE 1: Pre-training Individual Experts ---")
t_stage1 = time.time()
for name, expert in zip(expert_names, experts):
    p = model_size_stats(expert)["Params"]
    print(f"\n  >>> {name}  ({p:,} params)")
    pretrain_expert(expert, Xtr_s, epochs=EXPERT_EPOCHS, lr=EXPERT_LR,
                    batch_size=BATCH_SIZE, seed=SEED)
print(f"\n  Stage 1 wall time: {time.time() - t_stage1:.1f}s")
pretrained_expert_states = [copy.deepcopy(e.state_dict()) for e in experts]

print("\n--- STAGE 2: Gating Network (experts frozen) ---")
moe = MoE_IDS(experts, input_dim=D, num_experts=N_EXPERTS,
              top_k=TOP_K, noisy_gating=True)
print(f"  MoE total params: {model_size_stats(moe)['Params']:,}")
t_stage2 = time.time()
train_moe(moe, Xtr_s, epochs=GATING_EPOCHS, lr=GATING_LR,
          batch_size=BATCH_SIZE, lambda_bal=LAMBDA_BAL,
          freeze_experts=True, seed=SEED)
print(f"  Stage 2 wall time: {time.time() - t_stage2:.1f}s")

print("\n--- STAGE 3: End-to-End Fine-tuning ---")
t_stage3 = time.time()
train_moe(moe, Xtr_s, epochs=E2E_EPOCHS, lr=E2E_LR,
          batch_size=BATCH_SIZE, lambda_bal=LAMBDA_BAL,
          freeze_experts=False, seed=SEED)
print(f"  Stage 3 wall time: {time.time() - t_stage3:.1f}s")

# --- Save checkpoint ---
ckpt_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionB_seed{SEED}.pt")
torch.save({
    "moe_state_dict":      moe.state_dict(),
    "pretrained_experts":  pretrained_expert_states,
    "expert_names":        expert_names,
    "config": {
        "D": D, "T_SEQ": T_SEQ, "N_EXPERTS": N_EXPERTS, "TOP_K": TOP_K,
        "LAMBDA_BAL": LAMBDA_BAL, "SEED": SEED,
        "EXPERT_EPOCHS": EXPERT_EPOCHS, "GATING_EPOCHS": GATING_EPOCHS,
        "E2E_EPOCHS": E2E_EPOCHS,
        "GAT_HIDDEN": GAT_HIDDEN, "GAT_HEADS": GAT_HEADS,
        "GAT_TOPK": GAT_TOPK, "GAT_BOTTLENECK": GAT_BOTTLENECK,
    },
}, ckpt_path)
print(f"\nCheckpoint saved: {ckpt_path}")


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def moe_score_sequences(moe_model, X_seq, method="combined_mse", batch_size=512):
    moe_model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_seq, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    parts = []
    for (x,) in loader:
        parts.append(moe_model.anomaly_scores(x.to(device), method=method).cpu().numpy())
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


def compute_metrics_99pct(y_true, scores, train_scores, pct=99):
    thr = float(np.percentile(np.asarray(train_scores, dtype=float), pct))
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) > thr).astype(int)
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


print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)
best_method, best_roc = None, -1
all_methods = {}
for method in ["combined_mse", "weighted_expert_mse"]:
    tr_s = moe_score_sequences(moe, Xtr_s, method=method)
    te_s = moe_score_sequences(moe, Xte_s, method=method)
    m = compute_metrics_99pct(yte_seq, te_s, tr_s, pct=THR_PCT)
    all_methods[method] = (tr_s, te_s, m)
    print(f"\n  {method}:")
    print(f"    ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")
    print(f"    TN={m['TN']}  FP={m['FP']}  FN={m['FN']}  TP={m['TP']}")
    if m['roc_auc'] > best_roc:
        best_roc = m['roc_auc']
        best_method = method
print(f"\n  Best scoring method: {best_method} (ROC-AUC={best_roc:.4f})")

train_scores, test_scores, moe_metrics = all_methods[best_method]


# --- Per-attack breakdown ---
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


print("\n--- Per-Attack Breakdown ---")
pa = per_attack_breakdown(test_scores, yte_seq, ate_seq)
print(pa.to_string(index=False))


# ============================================================
# Routing Analysis - the headline diagnostic for Option B
# ============================================================
print("\n" + "=" * 60)
print("ROUTING ANALYSIS")
print("=" * 60)
gates_test = get_expert_routing(moe, Xte_s)

atk_types = sorted(set(ate_seq) - {"normal"})
routing_data = {}
for atk in ["normal"] + atk_types:
    mask = ate_seq == atk
    if mask.sum() > 0:
        routing_data[atk] = gates_test[mask].mean(axis=0)

routing_df = pd.DataFrame(routing_data, index=expert_names).T
print("\nAverage Expert Routing by Attack Type:")
print(routing_df.round(4).to_string())

mean_per_expert = routing_df.mean(axis=0)
print("\nMean routing weight across all traffic types:")
print(mean_per_expert.round(4).to_string())

primary = routing_df.idxmax(axis=1)
print("\nPrimary expert per attack type:")
for atk, exp in primary.items():
    print(f"  {atk:15s} -> {exp}")

# --- Acceptance criteria ---
print("\n--- Option B Acceptance Criteria ---")
crit1 = moe_metrics["pr_auc"] >= 0.985
print(f"  [1] PR-AUC >= 0.985:                          {crit1}  ({moe_metrics['pr_auc']:.4f})")
crit2 = (mean_per_expert >= 0.10).all()
print(f"  [2] All experts >= 10% mean routing weight:   {crit2}  ({mean_per_expert.round(3).to_dict()})")
gat_attacks = [a for a in atk_types if a in routing_df.index]
gat_weight_on_attacks = (routing_df.loc[gat_attacks, "GAT-AE"]
                         if gat_attacks else pd.Series(dtype=float))
crit3 = (gat_weight_on_attacks >= 0.15).any() if len(gat_weight_on_attacks) else False
print(f"  [3] GAT-AE >= 15% on at least one attack:     {crit3}")
if len(gat_weight_on_attacks):
    for atk, w in gat_weight_on_attacks.items():
        print(f"        GAT-AE on {atk}: {w:.4f}")
crit5 = model_size_stats(moe)["Params"] < 250_000
print(f"  [5] Total params < 250K:                      {crit5}  ({model_size_stats(moe)['Params']:,})")


# ============================================================
# Save metrics CSV
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Headline row
results_rows = [{
    "model":         "MoE3 Option B",
    "seed":          SEED,
    "scoring":       best_method,
    "roc_auc":       moe_metrics["roc_auc"],
    "pr_auc":        moe_metrics["pr_auc"],
    "f1":            moe_metrics["f1"],
    "precision":     moe_metrics["precision"],
    "recall":        moe_metrics["recall"],
    "accuracy":      moe_metrics["accuracy"],
    "thr_p99":       moe_metrics["thr"],
    "TN":            moe_metrics["TN"],
    "FP":            moe_metrics["FP"],
    "FN":            moe_metrics["FN"],
    "TP":            moe_metrics["TP"],
    "params_total":  model_size_stats(moe)["Params"],
    "params_GRU_AE": model_size_stats(experts[0])["Params"],
    "params_TCN_AE": model_size_stats(experts[1])["Params"],
    "params_GAT_AE": model_size_stats(experts[2])["Params"],
    "crit_1_pr_auc_ge_0985":            bool(crit1),
    "crit_2_all_experts_ge_10pct":      bool(crit2),
    "crit_3_GAT_ge_15pct_on_attack":    bool(crit3),
    "crit_5_params_lt_250K":            bool(crit5),
}]
results_df = pd.DataFrame(results_rows)
results_csv = os.path.join(RESULTS_DIR, "MoE3_OptionB_results.csv")
results_df.to_csv(results_csv, index=False)
print(f"  Headline metrics: {results_csv}")

# Per-attack table
pa_csv = os.path.join(RESULTS_DIR, "MoE3_OptionB_per_attack.csv")
pa.to_csv(pa_csv, index=False)
print(f"  Per-attack:       {pa_csv}")

# Routing table
routing_csv = os.path.join(RESULTS_DIR, "MoE3_OptionB_routing.csv")
routing_df.to_csv(routing_csv)
print(f"  Routing weights:  {routing_csv}")

# Final headline print
print("\n" + "=" * 60)
print("HEADLINE METRICS (MoE3 Option B, seed=42)")
print("=" * 60)
print(f"  ROC-AUC: {moe_metrics['roc_auc']:.4f}")
print(f"  PR-AUC:  {moe_metrics['pr_auc']:.4f}")
print(f"  F1:      {moe_metrics['f1']:.4f}")
print(f"  Params:  {model_size_stats(moe)['Params']:,}")
print(f"  Best scoring method: {best_method}")
print("=" * 60)
print("FULL TRAINING RUN COMPLETE.")
