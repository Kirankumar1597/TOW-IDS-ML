"""Standalone smoke-test driver for MoE3-IDS Option B.

Imports/inlines the same code that lives in the notebook cells, runs a 1-epoch
dry pass on a 100-sample subset, and prints the verdict. Used to verify
shape correctness and parameter counts before launching full training.
"""
import os, sys, random, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Config (mirrors the notebook config cell)
# ============================================================
OUT_DIR = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/extracted"
y_train_path = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_train.csv"
y_test_path  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_test.csv"

WINDOW_SEC = 1.0
T_SEQ      = 40
THR_PCT    = 99

N_EXPERTS  = 3
TOP_K      = 2
LAMBDA_BAL = 0.01
EXPERT_LR  = 1e-3
GATING_LR  = 1e-3
E2E_LR     = 1e-4

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
# Data ingestion + windowing (verbatim from notebook)
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

for df in [train_pkts]:
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
    seqs, seq_y = [], []
    for start in range(0, len(df) - t_seq + 1, stride):
        seqs.append(X[start:start + t_seq])
        seq_y.append(int(y[start:start + t_seq].max()))
    return np.stack(seqs), np.array(seq_y)


print("[data] Building windows ...")
train_w = build_global_windows_with_attack(train_pkts, window_sec=WINDOW_SEC)
feat_cols = get_feature_cols(train_w)
D = len(feat_cols)
Xtr_seq, _ = make_sequences(
    train_w[train_w["y_window"] == 0].reset_index(drop=True),
    t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
scaler = StandardScaler().fit(Xtr_seq.reshape(-1, D))
Xtr_s = scaler.transform(Xtr_seq.reshape(-1, D)).reshape(Xtr_seq.shape).astype(np.float32)
print(f"[data] D={D}, Xtr_s={Xtr_s.shape}")


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
# Smoke Test (1 epoch, 100 samples, all 3 stages)
# ============================================================
print("\n" + "=" * 60)
print("SMOKE TEST - 1 epoch, 100 samples, all three stages")
print("=" * 60)

set_seed(42)
SMOKE_N = 100
SMOKE_EPOCHS = 1
SMOKE_BATCH = 32

spec_params = {"GRU-AE": 65_000, "TCN-AE": 102_000, "GAT-AE": 35_000}
smoke_experts = [
    GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2),
    TCN_AE_Expert(D, channels=64, kernel_size=3),
    GAT_AE_Expert(D=D, T=T_SEQ, hidden=GAT_HIDDEN, heads=GAT_HEADS,
                  top_k=GAT_TOPK, bottleneck=GAT_BOTTLENECK),
]
smoke_names = ["GRU-AE", "TCN-AE", "GAT-AE"]

print("\n[1] Parameter counts vs spec target:")
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

print("\n[2] Shape check on a (4, 40, 12) batch:")
x_dummy = torch.randn(4, T_SEQ, D, device=device)
for name, exp in zip(smoke_names, smoke_experts):
    exp = exp.to(device).eval()
    with torch.no_grad():
        y_dummy = exp(x_dummy)
    ok = tuple(y_dummy.shape) == (4, T_SEQ, D)
    print(f"  [{'OK' if ok else 'FAIL':4s}] {name:7s}  in={tuple(x_dummy.shape)}  out={tuple(y_dummy.shape)}")
    if not ok:
        raise RuntimeError(f"{name} reconstruction shape mismatch")

Xtr_smoke = Xtr_s[:SMOKE_N]
print(f"\n[3] Stage 1: pre-train each expert for {SMOKE_EPOCHS} epoch on {SMOKE_N} samples")
for name, exp in zip(smoke_names, smoke_experts):
    print(f"  --- {name} ---")
    pretrain_expert(exp, Xtr_smoke, epochs=SMOKE_EPOCHS, lr=EXPERT_LR,
                    batch_size=SMOKE_BATCH, seed=42)

print(f"\n[4] Stage 2: gating-only training for {SMOKE_EPOCHS} epoch")
moe_smoke = MoE_IDS(smoke_experts, input_dim=D, num_experts=N_EXPERTS,
                    top_k=TOP_K, noisy_gating=True)
print(f"  MoE total params: {model_size_stats(moe_smoke)['Params']:,}")
train_moe(moe_smoke, Xtr_smoke, epochs=SMOKE_EPOCHS, lr=GATING_LR,
          batch_size=SMOKE_BATCH, lambda_bal=LAMBDA_BAL,
          freeze_experts=True, seed=42)

print(f"\n[5] Stage 3: end-to-end fine-tuning for {SMOKE_EPOCHS} epoch")
train_moe(moe_smoke, Xtr_smoke, epochs=SMOKE_EPOCHS, lr=E2E_LR,
          batch_size=SMOKE_BATCH, lambda_bal=LAMBDA_BAL,
          freeze_experts=False, seed=42)

print("\n[6] End-to-end MoE forward shape check:")
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

print("\n" + "=" * 60)
warns = [d for d in deviations if abs(d[3]) > 10.0]
if warns:
    print(f"SMOKE TEST FINISHED with {len(warns)} parameter-count deviation(s) >10%:")
    for name, p, target, dev in warns:
        print(f"   - {name}: {p:,} vs target {target:,} ({dev:+.1f}%)")
    print("Per spec: review before launching full training.")
else:
    print("SMOKE TEST PASSED. All shapes correct, all params within 10% of spec.")
print("=" * 60)
