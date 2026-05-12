"""D2.5c — Score-distribution overlap analysis (paper-Figure-1 candidate).

For each of GRU-AE, TCN-AE, GAT-AE (standalone) and the seed-42 MoE:
  - Compute anomaly scores on train-normal, test-normal, test-attack pools.
  - Compute the AUC of train-normal vs test-normal (i.e., how well the model's
    score function separates train-normal from test-normal — a direct measure
    of the distribution shift seen by that model).
  - AUC near 0.5 -> train and test normals indistinguishable (no shift seen).
  - AUC near 1.0 -> model treats test-normals as systematically different from
    train-normals (large shift seen).

Outputs:
  - results/MoE3_OptionB_normal_vs_normal_auc.csv
  - results/MoE3_OptionB_score_distributions.png
       4 expert blocks arranged in a 2x2 grand-grid; each block is a 2x2
       inner panel showing (top-L) train-normal histogram, (top-R) test-normal
       histogram, (bot-L) test-attack histogram, (bot-R) overlay of all three.

No retraining; uses checkpoints/MoE3_OptionB_seed42.pt.
"""
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

WORKDIR        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(WORKDIR, "checkpoints")
RESULTS_DIR    = os.path.join(WORKDIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Config — must match seed-42 training run exactly
# ============================================================
OUT_DIR      = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/extracted"
y_train_path = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_train.csv"
y_test_path  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_test.csv"

WINDOW_SEC = 1.0
T_SEQ      = 40
SEED       = 42

N_EXPERTS  = 3
TOP_K      = 2

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
# Data ingestion (matches full_train_moe3.py)
# ============================================================
print("\n[data] loading cached packet CSVs ...")
train_pkts = pd.read_csv(f"{OUT_DIR}/packets_train.csv")
test_pkts  = pd.read_csv(f"{OUT_DIR}/packets_test.csv")
y_train = pd.read_csv(y_train_path, header=None,
                      names=["sample_number", "normal_or_abnormal", "attack_type"])
y_test  = pd.read_csv(y_test_path,  header=None,
                      names=["sample_number", "normal_or_abnormal", "attack_type"])
for pkts, labels in [(train_pkts, y_train), (test_pkts, y_test)]:
    pkts["frame.number"]   = pd.to_numeric(pkts["frame.number"], errors="coerce").astype("Int64")
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
    seqs, seq_y = [], []
    for start in range(0, len(df) - t_seq + 1, stride):
        seqs.append(X[start:start + t_seq])
        seq_y.append(int(y[start:start + t_seq].max()))
    return np.stack(seqs), np.array(seq_y)


print("[data] building windowed features ...")
train_w = build_global_windows_with_attack(train_pkts, WINDOW_SEC)
test_w  = build_global_windows_with_attack(test_pkts,  WINDOW_SEC)
feat_cols = get_feature_cols(train_w)
D = len(feat_cols)
Xtr_seq, _ = make_sequences(
    train_w[train_w["y_window"] == 0].reset_index(drop=True),
    t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
Xte_seq, yte_seq = make_sequences(
    test_w, t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
scaler = StandardScaler().fit(Xtr_seq.reshape(-1, D))
Xtr_s = scaler.transform(Xtr_seq.reshape(-1, D)).reshape(Xtr_seq.shape).astype(np.float32)
Xte_s = scaler.transform(Xte_seq.reshape(-1, D)).reshape(Xte_seq.shape).astype(np.float32)
print(f"[data] D={D}, Xtr_s={Xtr_s.shape}, Xte_s={Xte_s.shape}")
print(f"       test normals = {int((yte_seq==0).sum())}, test attacks = {int((yte_seq==1).sum())}")


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


# ============================================================
# Load checkpoint
# ============================================================
ckpt_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionB_seed{SEED}.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
print(f"\n[ckpt] loaded {ckpt_path}")

gru = GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2).to(device)
tcn = TCN_AE_Expert(D, channels=64, kernel_size=3).to(device)
gat = GAT_AE_Expert(D=D, T=T_SEQ).to(device)
gru.load_state_dict(ckpt["pretrained_experts"][0])
tcn.load_state_dict(ckpt["pretrained_experts"][1])
gat.load_state_dict(ckpt["pretrained_experts"][2])

moe = MoE_IDS([
    GRU_AE_Expert(D, hidden=64, proj=128, dropout=0.2),
    TCN_AE_Expert(D, channels=64, kernel_size=3),
    GAT_AE_Expert(D=D, T=T_SEQ),
], input_dim=D, num_experts=N_EXPERTS, top_k=TOP_K, noisy_gating=True).to(device)
moe.load_state_dict(ckpt["moe_state_dict"])


# ============================================================
# Score helpers
# ============================================================
@torch.no_grad()
def score_expert(expert, X, batch=512):
    expert.eval()
    parts = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch, shuffle=False)
    for (x,) in loader:
        x = x.to(device)
        parts.append(torch.mean((x - expert(x)) ** 2, dim=(1, 2)).cpu().numpy())
    return np.concatenate(parts)


@torch.no_grad()
def score_moe(moe_model, X, method="weighted_expert_mse", batch=512):
    moe_model.eval()
    parts = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch, shuffle=False)
    for (x,) in loader:
        parts.append(moe_model.anomaly_scores(x.to(device), method=method).cpu().numpy())
    return np.concatenate(parts)


# ============================================================
# Compute scores for all 4 models on three pools
# ============================================================
print("\n[scoring] computing all scores ...")
Xte_normal = Xte_s[yte_seq == 0]
Xte_attack = Xte_s[yte_seq == 1]

scoring = {
    "GRU-AE": (lambda X: score_expert(gru, X)),
    "TCN-AE": (lambda X: score_expert(tcn, X)),
    "GAT-AE": (lambda X: score_expert(gat, X)),
    "MoE":    (lambda X: score_moe(moe, X, method="weighted_expert_mse")),
}

scores = {name: {
    "train_normal": fn(Xtr_s),
    "test_normal":  fn(Xte_normal),
    "test_attack":  fn(Xte_attack),
} for name, fn in scoring.items()}


# ============================================================
# Compute train-vs-test-normal AUC per model
# Reference quantities also computed for the figure / paper context.
# ============================================================
def auc_binary(neg_scores, pos_scores):
    y = np.concatenate([np.zeros(len(neg_scores)), np.ones(len(pos_scores))])
    s = np.concatenate([neg_scores, pos_scores])
    return float(roc_auc_score(y, s))


rows = []
for name in scoring:
    sc = scores[name]
    auc_shift  = auc_binary(sc["train_normal"], sc["test_normal"])      # higher -> more shift seen
    auc_attack = auc_binary(sc["test_normal"],  sc["test_attack"])      # detection performance
    auc_train_vs_attack = auc_binary(sc["train_normal"], sc["test_attack"])
    rows.append({
        "model":                       name,
        "n_train_normal":              int(len(sc["train_normal"])),
        "n_test_normal":               int(len(sc["test_normal"])),
        "n_test_attack":               int(len(sc["test_attack"])),
        "train_normal_mean":           float(np.mean(sc["train_normal"])),
        "train_normal_std":            float(np.std(sc["train_normal"])),
        "test_normal_mean":            float(np.mean(sc["test_normal"])),
        "test_normal_std":             float(np.std(sc["test_normal"])),
        "test_attack_mean":            float(np.mean(sc["test_attack"])),
        "test_attack_std":             float(np.std(sc["test_attack"])),
        "auc_train_vs_test_normal":    auc_shift,
        "auc_test_normal_vs_attack":   auc_attack,
        "auc_train_normal_vs_attack":  auc_train_vs_attack,
    })

df_auc = pd.DataFrame(rows)
auc_csv = os.path.join(RESULTS_DIR, "MoE3_OptionB_normal_vs_normal_auc.csv")
df_auc.to_csv(auc_csv, index=False)
print(f"\n[D2.5c] saved: {auc_csv}")

# Print table
print("\n  Train-normal vs Test-normal AUC (HIGHER => model sees MORE shift):")
print(df_auc[[
    "model",
    "auc_train_vs_test_normal",
    "auc_test_normal_vs_attack",
    "auc_train_normal_vs_attack",
    "train_normal_mean", "test_normal_mean", "test_attack_mean",
]].round(4).to_string(index=False))


# ============================================================
# Figure: 4 expert blocks, each a 2x2 inner panel (paper Figure 1 candidate)
# ============================================================
print("\n[fig] rendering paper-Figure-1 candidate ...")
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 10, "legend.fontsize": 8})

fig = plt.figure(figsize=(13, 11))
outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.32, hspace=0.42)

block_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
order = ["GRU-AE", "TCN-AE", "GAT-AE", "MoE"]

color_train  = "#377eb8"  # blue
color_test_n = "#4daf4a"  # green
color_test_a = "#e41a1c"  # red
nbins = 32

for (br, bc), name in zip(block_positions, order):
    sc = scores[name]
    s_tr = sc["train_normal"]
    s_tn = sc["test_normal"]
    s_ta = sc["test_attack"]

    # Common x range across the three histograms in this block
    all_s = np.concatenate([s_tr, s_tn, s_ta])
    lo = np.percentile(all_s, 0.5)
    hi = np.percentile(all_s, 99.5)
    bins = np.linspace(lo, hi, nbins + 1)

    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[br, bc],
                                              wspace=0.32, hspace=0.42)
    ax_tl = fig.add_subplot(inner[0, 0])
    ax_tr = fig.add_subplot(inner[0, 1])
    ax_bl = fig.add_subplot(inner[1, 0])
    ax_br = fig.add_subplot(inner[1, 1])

    ax_tl.hist(s_tr, bins=bins, color=color_train, alpha=0.85)
    ax_tl.set_title("train-normal", fontsize=9)
    ax_tl.set_xlabel("score (MSE)"); ax_tl.set_ylabel("count")

    ax_tr.hist(s_tn, bins=bins, color=color_test_n, alpha=0.85)
    ax_tr.set_title("test-normal", fontsize=9)
    ax_tr.set_xlabel("score (MSE)")

    ax_bl.hist(s_ta, bins=bins, color=color_test_a, alpha=0.85)
    ax_bl.set_title("test-attack", fontsize=9)
    ax_bl.set_xlabel("score (MSE)"); ax_bl.set_ylabel("count")

    # Overlay (density) — the headline panel for this expert
    ax_br.hist(s_tr, bins=bins, color=color_train,  alpha=0.55, density=True, label="train-normal")
    ax_br.hist(s_tn, bins=bins, color=color_test_n, alpha=0.55, density=True, label="test-normal")
    ax_br.hist(s_ta, bins=bins, color=color_test_a, alpha=0.55, density=True, label="test-attack")
    ax_br.set_title("overlay (density)", fontsize=9)
    ax_br.set_xlabel("score (MSE)"); ax_br.set_ylabel("density")
    ax_br.legend(loc="upper right", framealpha=0.85)

    # Block title with the AUC numbers — the entire point of this figure
    auc_shift = auc_binary(s_tr, s_tn)
    auc_attack = auc_binary(s_tn, s_ta)
    block_ax_label = (f"{name}    "
                      f"AUC(train-N vs test-N) = {auc_shift:.3f}   "
                      f"AUC(test-N vs test-A) = {auc_attack:.3f}")
    fig.text(0.25 + 0.5 * bc, 0.95 - 0.5 * br, block_ax_label,
             ha="center", va="bottom", fontsize=11, fontweight="bold")

fig.suptitle("MoE3-IDS Option B  reconstruction-score distributions per expert\n"
             "Train-normal vs test-normal AUC quantifies the covariate shift each model perceives",
             y=1.00, fontsize=12)

png_path = os.path.join(RESULTS_DIR, "MoE3_OptionB_score_distributions.png")
plt.savefig(png_path, dpi=140, bbox_inches="tight")
plt.close()
print(f"[fig] saved: {png_path}")

print("\n" + "=" * 60)
print("D2.5c complete.")
print("=" * 60)
