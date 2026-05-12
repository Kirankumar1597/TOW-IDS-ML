"""Steps 2 + 3 of the Option C 5-step protocol.

Step 2 — Stage 2 gating training (50 epochs, lr=1e-3, lambda=0.01, GAN frozen).
        Loss uses z-normalized scores. Per-epoch traces of routing weights
        and per-expert raw / z-score statistics. Output:
          results/MoE3_OptionC_stage2_traces.csv

Step 3 — Stage 3 end-to-end fine-tuning (30 epochs, lr=1e-4, GAN frozen).
        Same trace logging + per-expert reconstruction loss and AUC(train-N
        vs test-N) at the final epoch. Output:
          results/MoE3_OptionC_stage3_traces.csv

Routing-weight evolution plot covers both stages on a single 80-epoch axis.
Loads the LOCKED checkpoint (immutable Stage-1 state).
"""
import os, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

WORKDIR        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(WORKDIR, "checkpoints")
RESULTS_DIR    = os.path.join(WORKDIR, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Config
# ============================================================
OUT_DIR      = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/extracted"
y_train_path = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_train.csv"
y_test_path  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_test.csv"

WINDOW_SEC = 1.0; T_SEQ = 40; SEED = 42
N_EXPERTS = 3; TOP_K = 2; LAMBDA_BAL = 0.01
GATING_EPOCHS = 50; GATING_LR = 1e-3
E2E_EPOCHS    = 30; E2E_LR    = 1e-4
BATCH_SIZE    = 256

DIFF_T_TOTAL=100; DIFF_T_EVAL=50; DIFF_BETA_LO=1e-4; DIFF_BETA_HI=0.02
DIFF_TIME_DIM=32; DIFF_CHANNELS=(32, 64)
GAN_LATENT = 16
SCORE_SIGMA_FLOOR = 0.05


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def get_feature_cols(df):
    return [c for c in df.columns if c not in ["w","y_window","attack_type_window"]]


def model_size_stats(m):
    return {"Params": int(sum(p.numel() for p in m.parameters()))}


# ============================================================
# Data ingestion (verbatim)
# ============================================================
print("\n[data] loading cached packet CSVs ...")
train_pkts = pd.read_csv(f"{OUT_DIR}/packets_train.csv")
test_pkts  = pd.read_csv(f"{OUT_DIR}/packets_test.csv")
y_train = pd.read_csv(y_train_path, header=None, names=["sample_number","normal_or_abnormal","attack_type"])
y_test  = pd.read_csv(y_test_path,  header=None, names=["sample_number","normal_or_abnormal","attack_type"])
for pkts, labels in [(train_pkts, y_train), (test_pkts, y_test)]:
    pkts["frame.number"]   = pd.to_numeric(pkts["frame.number"], errors="coerce").astype("Int64")
    labels["sample_number"] = pd.to_numeric(labels["sample_number"], errors="coerce").astype("Int64")
train_pkts = train_pkts.merge(y_train, left_on="frame.number", right_on="sample_number", how="left").drop(columns=["sample_number"])
train_pkts["y"] = (train_pkts["normal_or_abnormal"].str.lower() == "abnormal").astype(int)
test_pkts = test_pkts.merge(y_test, left_on="frame.number", right_on="sample_number", how="left").drop(columns=["sample_number"])
test_pkts["y"] = (test_pkts["normal_or_abnormal"].str.lower() == "abnormal").astype(int)
for df in [train_pkts, test_pkts]:
    at_col = None
    for c in ["attack_type","attack_type_y","attack_type_x"]:
        if c in df.columns and df[c].notna().any(): at_col = c; break
    if at_col and at_col != "attack_type": df["attack_type"] = df[at_col]
    df["attack_type"] = df["attack_type"].astype(str).str.strip().str.lower()


def build_global_windows_with_attack(df, window_sec=1.0):
    df = df.copy()
    df["t"] = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t")
    t0 = df["t"].iloc[0]
    df["w"] = np.floor((df["t"] - t0) / window_sec).astype(int)
    df["frame.len"] = pd.to_numeric(df["frame.len"], errors="coerce").fillna(0.0)
    df["dt"] = df["t"].diff().fillna(0.0).clip(lower=0.0)
    df["is_multicast"] = df["eth.dst"].astype(str).str.lower().str.startswith(("01:","33:")).astype(int)
    df["has_vlan"] = df["vlan.id"].notna().astype(int)
    def agg(g):
        y_win = int(pd.to_numeric(g["y"], errors="coerce").fillna(0).max())
        atk = "normal"
        if y_win == 1:
            atk_s = g.loc[g["y"]==1,"attack_type"].astype(str).str.strip()
            atk = atk_s.value_counts().index[0] if len(atk_s) else "anomaly"
        return pd.Series({
            "pkt_count": len(g), "bytes_sum": g["frame.len"].sum(),
            "pkt_len_mean": g["frame.len"].mean(), "pkt_len_std": g["frame.len"].std(ddof=0),
            "dt_mean": g["dt"].mean(), "dt_std": g["dt"].std(ddof=0),
            "uniq_src_mac": g["eth.src"].nunique(dropna=True),
            "uniq_dst_mac": g["eth.dst"].nunique(dropna=True),
            "uniq_ip_src": g["ip.src"].nunique(dropna=True),
            "uniq_ip_dst": g["ip.dst"].nunique(dropna=True),
            "multicast_ratio": g["is_multicast"].mean(), "vlan_ratio": g["has_vlan"].mean(),
            "y_window": y_win, "attack_type_window": atk,
        })
    return df.groupby("w").apply(agg).reset_index().fillna(0.0)


def make_sequences(df, t_seq=20, stride=1, feature_cols=None):
    df = df.sort_values("w").reset_index(drop=True)
    if feature_cols is None: feature_cols = get_feature_cols(df)
    X = df[feature_cols].astype(float).to_numpy()
    y = df["y_window"].astype(int).to_numpy()
    seqs, seq_y = [], []
    for s in range(0, len(df)-t_seq+1, stride):
        seqs.append(X[s:s+t_seq]); seq_y.append(int(y[s:s+t_seq].max()))
    return np.stack(seqs), np.array(seq_y)


train_w = build_global_windows_with_attack(train_pkts, WINDOW_SEC)
test_w  = build_global_windows_with_attack(test_pkts,  WINDOW_SEC)
feat_cols = get_feature_cols(train_w); D = len(feat_cols)
Xtr_seq, _ = make_sequences(train_w[train_w["y_window"]==0].reset_index(drop=True),
                              t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
Xte_seq, yte_seq = make_sequences(test_w, t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
scaler = StandardScaler().fit(Xtr_seq.reshape(-1, D))
Xtr_s = scaler.transform(Xtr_seq.reshape(-1, D)).reshape(Xtr_seq.shape).astype(np.float32)
Xte_s = scaler.transform(Xte_seq.reshape(-1, D)).reshape(Xte_seq.shape).astype(np.float32)
print(f"[data] D={D}, Xtr_s={Xtr_s.shape}, Xte_s={Xte_s.shape}")


# ============================================================
# Architectures (mirror; includes anomaly_score on each)
# ============================================================
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
    def anomaly_score(self, x):
        return torch.mean((x - self.forward(x)) ** 2, dim=(1, 2))


class TS_Generator(nn.Module):
    def __init__(self, latent=GAN_LATENT, T=T_SEQ, D=12):
        super().__init__()
        self.T, self.D = T, D
        self.fc = nn.Sequential(
            nn.Linear(latent, 128),  nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),     nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, T * D),
        )
    def forward(self, z): return self.fc(z).view(-1, self.T, self.D)


class TS_Discriminator(nn.Module):
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
        h = self.conv(x_seq.transpose(1, 2))
        return self.fc(h.squeeze(-1))


class GAN_Expert(nn.Module):
    def __init__(self, latent=GAN_LATENT, T=T_SEQ, D=12, ch=64):
        super().__init__()
        self.G    = TS_Generator(latent, T, D)
        self.Disc = TS_Discriminator(T, D, ch)
        self.latent = latent
    def forward(self, x_seq): return self.Disc(x_seq).squeeze(-1)
    def anomaly_score(self, x_seq):
        return 1.0 - self.Disc(x_seq).squeeze(-1)


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim=DIFF_TIME_DIM):
        super().__init__()
        self.time_dim = time_dim
        self.mlp = nn.Sequential(nn.Linear(time_dim, time_dim*2), nn.SiLU(),
                                  nn.Linear(time_dim*2, time_dim))
    def forward(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(-np.log(10000.0) *
                           torch.arange(half, device=t.device).float() / max(half, 1))
        emb = t.float()[:, None] * freqs[None, :]
        return self.mlp(torch.cat([emb.sin(), emb.cos()], dim=-1))


class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(max(1, min(8, in_c)), in_c)
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_c)
        self.norm2 = nn.GroupNorm(max(1, min(8, out_c)), out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Diffusion1DUNet(nn.Module):
    def __init__(self, D=12, channels=DIFF_CHANNELS, time_dim=DIFF_TIME_DIM):
        super().__init__()
        ch_mid, ch_max = channels
        self.time_emb = TimeEmbedding(time_dim)
        self.in_proj  = nn.Conv1d(D, ch_mid, 3, padding=1)
        self.down1    = ResBlock1D(ch_mid, ch_mid, time_dim)
        self.down2    = ResBlock1D(ch_mid, ch_max, time_dim)
        self.mid      = ResBlock1D(ch_max, ch_max, time_dim)
        self.up1      = ResBlock1D(ch_max+ch_max, ch_mid, time_dim)
        self.up2      = ResBlock1D(ch_mid+ch_mid, ch_mid, time_dim)
        self.out_proj = nn.Conv1d(ch_mid, D, 3, padding=1)
    def forward(self, x_seq, t):
        x = x_seq.transpose(1, 2); t_emb = self.time_emb(t)
        h0 = self.in_proj(x); h1 = self.down1(h0, t_emb); h2 = self.down2(h1, t_emb)
        h = self.mid(h2, t_emb)
        h = self.up1(torch.cat([h, h2], dim=1), t_emb)
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)
        return self.out_proj(h).transpose(1, 2)


class DDPM_Expert(nn.Module):
    def __init__(self, D=12, T=T_SEQ, channels=DIFF_CHANNELS,
                 t_total=DIFF_T_TOTAL, t_eval=DIFF_T_EVAL,
                 beta_lo=DIFF_BETA_LO, beta_hi=DIFF_BETA_HI,
                 time_dim=DIFF_TIME_DIM, eval_seed=42):
        super().__init__()
        self.D, self.T = D, T; self.t_total = t_total; self.t_eval = t_eval
        self.eval_seed = eval_seed
        self.unet = Diffusion1DUNet(D=D, channels=channels, time_dim=time_dim)
        betas = torch.linspace(beta_lo, beta_hi, t_total)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))
    def add_noise(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        ab = self.alpha_bars[t]
        if ab.dim() == 0: x_t = ab.sqrt()*x_0 + (1-ab).sqrt()*noise
        else: ab = ab[:, None, None]; x_t = ab.sqrt()*x_0 + (1-ab).sqrt()*noise
        return x_t, noise
    def forward(self, x_seq):
        B = x_seq.size(0)
        t = torch.randint(0, self.t_total, (B,), device=x_seq.device)
        x_t, noise = self.add_noise(x_seq, t)
        return self.unet(x_t, t), noise
    def anomaly_score(self, x_seq):
        B = x_seq.size(0)
        gen = torch.Generator(device=x_seq.device).manual_seed(self.eval_seed)
        noise = torch.empty_like(x_seq).normal_(generator=gen)
        t = torch.full((B,), self.t_eval, device=x_seq.device, dtype=torch.long)
        x_t, true_noise = self.add_noise(x_seq, t, noise=noise)
        pred = self.unet(x_t, t)
        return torch.mean((pred - true_noise) ** 2, dim=(1, 2))


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


class MoE_OptionC(nn.Module):
    SCORE_SIGMA_FLOOR_ = 0.05  # used in Step 1 calibration; floor already locked in buffers.

    def __init__(self, experts, input_dim, num_experts=3, top_k=2, noisy_gating=True):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating  = SparseGatingNetwork(input_dim, num_experts, top_k, noisy_gating)
        self.num_experts = num_experts; self.top_k = top_k
        self.register_buffer("score_mu",    torch.zeros(num_experts))
        self.register_buffer("score_sigma", torch.ones(num_experts))

    def forward(self, x):
        raw_scores = torch.stack([e.anomaly_score(x) for e in self.experts], dim=1)
        z_scores = (raw_scores - self.score_mu) / self.score_sigma
        gates, raw_logits = self.gating(x)
        anomaly = (gates * z_scores).sum(dim=1)
        return anomaly, gates, raw_logits, raw_scores, z_scores

    def anomaly_scores(self, x, method="z_combined"):
        anomaly, gates, _, raw_scores, z_scores = self.forward(x)
        if method == "z_combined": return anomaly
        if method == "raw_combined": return (gates * raw_scores).sum(dim=1)
        if method == "max_z": return z_scores.max(dim=1).values
        return anomaly


def load_balancing_loss(gates, raw_logits, num_experts):
    f = gates.mean(dim=0)
    P = F.softmax(raw_logits, dim=-1).mean(dim=0)
    return num_experts * (f * P).sum()


# ============================================================
# Helpers
# ============================================================
@torch.no_grad()
def compute_traces(moe, X, expert_names):
    """Forward over X (batched), gather per-expert raw/z-score stats and avg gates."""
    moe_was_training = moe.training
    moe.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=512, shuffle=False)
    all_raw, all_z, all_gates = [], [], []
    for (x,) in loader:
        x = x.to(device)
        _, gates, _, raw, z = moe(x)
        all_raw.append(raw.cpu().numpy())
        all_z.append(z.cpu().numpy())
        all_gates.append(gates.cpu().numpy())
    raw   = np.concatenate(all_raw,   axis=0)
    z     = np.concatenate(all_z,     axis=0)
    gates = np.concatenate(all_gates, axis=0)
    out = {"gate_mean": gates.mean(axis=0)}                # (E,)
    for i, name in enumerate(expert_names):
        out[f"raw_mean_{name}"] = float(raw[:, i].mean())
        out[f"raw_std_{name}"]  = float(raw[:, i].std())
        out[f"z_mean_{name}"]   = float(z[:, i].mean())
        out[f"z_std_{name}"]    = float(z[:, i].std())
        out[f"gate_mean_{name}"] = float(gates[:, i].mean())
    if moe_was_training:
        moe.train()
    return out


@torch.no_grad()
def expert_scores_array(expert, X, batch=512):
    expert.eval()
    parts = []
    for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch, shuffle=False):
        parts.append(expert.anomaly_score(x.to(device)).cpu().numpy())
    return np.concatenate(parts)


def auc_train_vs_test_normal(scores_train_normal, scores_test_normal):
    a = np.asarray(scores_train_normal, dtype=float)
    b = np.asarray(scores_test_normal, dtype=float)
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    s = np.concatenate([a, b])
    return float(roc_auc_score(y, s))


# ============================================================
# Load LOCKED Stage 1 checkpoint
# ============================================================
locked_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionC_stage1_seed{SEED}_LOCKED.pt")
print(f"\n[locked] loading: {locked_path}")
locked = torch.load(locked_path, map_location=device, weights_only=False)
expert_names = list(locked["expert_names"])
print(f"[locked] expert_names = {expert_names}")
print(f"[locked] mu    = {locked['score_mu'].cpu().numpy().round(4)}")
print(f"[locked] sigma = {locked['score_sigma'].cpu().numpy().round(4)}")
print(f"[locked] sigma_measured (pre-floor) = {locked['score_sigma_measured'].cpu().numpy().round(4)}")

tcn  = TCN_AE_Expert(D, channels=64, kernel_size=3).to(device)
gan  = GAN_Expert(latent=GAN_LATENT, T=T_SEQ, D=D, ch=64).to(device)
ddpm = DDPM_Expert(D=D, T=T_SEQ, channels=DIFF_CHANNELS).to(device)
tcn.load_state_dict(locked["tcn_state"])
gan.load_state_dict(locked["gan_state"])
ddpm.load_state_dict(locked["ddpm_state"])

experts = [tcn, gan, ddpm]
moe = MoE_OptionC(experts, input_dim=D, num_experts=N_EXPERTS,
                   top_k=TOP_K, noisy_gating=True).to(device)
moe.score_mu.copy_(locked["score_mu"].to(device))
moe.score_sigma.copy_(locked["score_sigma"].to(device))
print(f"[moe] total params: {model_size_stats(moe)['Params']:,}")


# ============================================================
# Step 2 — Stage 2 gating (50 epochs, lr=1e-3, GAN frozen, all experts frozen)
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 2 — Stage 2 gating training ({GATING_EPOCHS} epochs, lr={GATING_LR}, "
      f"lambda={LAMBDA_BAL}, all experts frozen)")
print("=" * 60)
set_seed(SEED)
for e in moe.experts:
    for p in e.parameters(): p.requires_grad = False
gate_params = list(moe.gating.parameters())
opt2 = optim.Adam(gate_params, lr=GATING_LR)
loader = DataLoader(TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32)),
                    batch_size=BATCH_SIZE, shuffle=True)

stage2_rows = []
t0 = time.time()
for ep in range(GATING_EPOCHS):
    moe.train()
    ep_anom, ep_bal = [], []
    for (x,) in loader:
        x = x.to(device)
        anomaly, gates, raw_logits, _, _ = moe(x)
        loss_anom = (anomaly ** 2).mean()
        loss_bal  = load_balancing_loss(gates, raw_logits, moe.num_experts)
        loss = loss_anom + LAMBDA_BAL * loss_bal
        opt2.zero_grad(); loss.backward(); opt2.step()
        ep_anom.append(loss_anom.item()); ep_bal.append(loss_bal.item())
    # Per-epoch trace (over full Xtr_s in eval mode)
    tr = compute_traces(moe, Xtr_s, expert_names)
    row = {
        "stage": 2, "epoch": ep + 1,
        "loss_anom": float(np.mean(ep_anom)),
        "loss_bal":  float(np.mean(ep_bal)),
    }
    row.update(tr)
    stage2_rows.append(row)
    if ep % 10 == 0 or ep == GATING_EPOCHS - 1:
        gw = tr["gate_mean"]
        print(f"  ep {ep+1:3d}: anom={row['loss_anom']:.4f}  bal={row['loss_bal']:.4f}  "
              f"gate=[{gw[0]:.3f}, {gw[1]:.3f}, {gw[2]:.3f}]")
print(f"Stage 2 wall time: {time.time()-t0:.1f}s")

stage2_df = pd.DataFrame(stage2_rows).drop(columns=["gate_mean"])  # gate_mean is array, exploded as gate_mean_*
stage2_csv = os.path.join(RESULTS_DIR, "MoE3_OptionC_stage2_traces.csv")
stage2_df.to_csv(stage2_csv, index=False)
print(f"Saved: {stage2_csv}")


# ============================================================
# Step 3 — Stage 3 e2e (30 epochs, lr=1e-4, GAN frozen)
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 3 — Stage 3 e2e fine-tuning ({E2E_EPOCHS} epochs, lr={E2E_LR}, "
      f"GAN frozen, TCN-AE + DDPM + gate trained)")
print("=" * 60)
set_seed(SEED)
# Unfreeze TCN-AE (idx 0) and DDPM (idx 2); GAN (idx 1) stays frozen.
for p in moe.experts[0].parameters(): p.requires_grad = True
for p in moe.experts[2].parameters(): p.requires_grad = True
for p in moe.experts[1].parameters(): p.requires_grad = False
params3 = [p for p in moe.parameters() if p.requires_grad]
opt3 = optim.Adam(params3, lr=E2E_LR)

stage3_rows = []
t0 = time.time()
for ep in range(E2E_EPOCHS):
    moe.train()
    ep_anom, ep_bal = [], []
    for (x,) in loader:
        x = x.to(device)
        anomaly, gates, raw_logits, _, _ = moe(x)
        loss_anom = (anomaly ** 2).mean()
        loss_bal  = load_balancing_loss(gates, raw_logits, moe.num_experts)
        loss = loss_anom + LAMBDA_BAL * loss_bal
        opt3.zero_grad(); loss.backward(); opt3.step()
        ep_anom.append(loss_anom.item()); ep_bal.append(loss_bal.item())
    tr = compute_traces(moe, Xtr_s, expert_names)
    row = {
        "stage": 3, "epoch": GATING_EPOCHS + ep + 1,  # combined-axis epoch index
        "stage3_epoch": ep + 1,
        "loss_anom": float(np.mean(ep_anom)),
        "loss_bal":  float(np.mean(ep_bal)),
    }
    row.update(tr)
    stage3_rows.append(row)
    if ep % 5 == 0 or ep == E2E_EPOCHS - 1:
        gw = tr["gate_mean"]
        print(f"  ep {ep+1:3d}: anom={row['loss_anom']:.4f}  bal={row['loss_bal']:.4f}  "
              f"gate=[{gw[0]:.3f}, {gw[1]:.3f}, {gw[2]:.3f}]")
print(f"Stage 3 wall time: {time.time()-t0:.1f}s")

stage3_df = pd.DataFrame(stage3_rows).drop(columns=["gate_mean"])
stage3_csv = os.path.join(RESULTS_DIR, "MoE3_OptionC_stage3_traces.csv")
stage3_df.to_csv(stage3_csv, index=False)
print(f"Saved: {stage3_csv}")


# ============================================================
# Final-epoch additions: per-expert reconstruction loss + AUC(train-N vs test-N)
# ============================================================
print("\n[final] computing per-expert reconstruction loss + AUC(train-N vs test-N) ...")
Xte_normal = Xte_s[yte_seq == 0]

# TCN-AE reconstruction MSE (its native Stage 1 objective)
tcn.eval()
with torch.no_grad():
    x_full = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    tcn_recon_mse = float(torch.mean((tcn(x_full) - x_full) ** 2).item())

# DDPM noise-prediction MSE (deterministic noise to keep it consistent across epochs)
ddpm.eval()
with torch.no_grad():
    s = expert_scores_array(ddpm, Xtr_s)
    ddpm_noise_mse_train = float(np.mean(s))

# GAN: report mean anomaly score on train normals (1 - D(x))
gan.eval()
with torch.no_grad():
    gan_train_score = float(np.mean(expert_scores_array(gan, Xtr_s)))

per_expert_recon = {
    "TCN-AE": tcn_recon_mse,
    "GAN":    gan_train_score,    # GAN's native loss is BCE-D; report 1-D(x) on train as analog
    "DDPM":   ddpm_noise_mse_train,
}

per_expert_auc_shift = {}
for name, exp in zip(expert_names, experts):
    s_tr = expert_scores_array(exp, Xtr_s)
    s_te = expert_scores_array(exp, Xte_normal)
    per_expert_auc_shift[name] = auc_train_vs_test_normal(s_tr, s_te)

# Also MoE-level scores
moe.eval()
def moe_scores(X, method="z_combined", batch=512):
    parts = []
    with torch.no_grad():
        for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                                batch_size=batch, shuffle=False):
            parts.append(moe.anomaly_scores(x.to(device), method=method).cpu().numpy())
    return np.concatenate(parts)

moe_tr = moe_scores(Xtr_s, method="z_combined")
moe_te_n = moe_scores(Xte_normal, method="z_combined")
moe_auc_shift = auc_train_vs_test_normal(moe_tr, moe_te_n)

print(f"\n  Per-expert recon-loss proxy (final epoch of Stage 3, on Xtr_s):")
for name, v in per_expert_recon.items():
    print(f"    {name:6s}: {v:.6f}")
print(f"\n  Per-expert AUC(train-N vs test-N):")
for name, v in per_expert_auc_shift.items():
    print(f"    {name:6s}: {v:.4f}")
print(f"  MoE          AUC(train-N vs test-N): {moe_auc_shift:.4f}")

# Save final-epoch summary CSV
finals_df = pd.DataFrame([{
    "model": name,
    "recon_loss_proxy": per_expert_recon[name],
    "auc_train_vs_test_normal": per_expert_auc_shift[name],
} for name in expert_names] + [{
    "model": "MoE", "recon_loss_proxy": np.nan,
    "auc_train_vs_test_normal": moe_auc_shift,
}])
finals_csv = os.path.join(RESULTS_DIR, "MoE3_OptionC_stage3_final.csv")
finals_df.to_csv(finals_csv, index=False)
print(f"Saved: {finals_csv}")


# ============================================================
# Save Stage 3 checkpoint
# ============================================================
ck3_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionC_stage3_seed{SEED}.pt")
torch.save({
    "moe_state_dict": moe.state_dict(),
    "expert_names": expert_names,
    "config": {
        "D": D, "T_SEQ": T_SEQ, "SEED": SEED, "TOP_K": TOP_K, "LAMBDA_BAL": LAMBDA_BAL,
        "GATING_EPOCHS": GATING_EPOCHS, "E2E_EPOCHS": E2E_EPOCHS,
        "DIFF_CHANNELS": DIFF_CHANNELS, "GAN_LATENT": GAN_LATENT,
        "DIFF_T_EVAL": DIFF_T_EVAL, "SCORE_SIGMA_FLOOR": SCORE_SIGMA_FLOOR,
    },
    "score_mu":    moe.score_mu.cpu(),
    "score_sigma": moe.score_sigma.cpu(),
}, ck3_path)
print(f"\n[stage3] checkpoint: {ck3_path}")


# ============================================================
# Routing-weight evolution plot (Stage 2 + Stage 3 on a single 80-epoch axis)
# ============================================================
print("\n[plot] rendering routing-weight evolution ...")
combined = pd.concat([stage2_df, stage3_df], ignore_index=True)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                          gridspec_kw={"height_ratios": [2.2, 1.0]})

# Top panel: per-expert mean gate weight per epoch
ax = axes[0]
colors = {"TCN-AE": "C0", "GAN": "C1", "DDPM": "C2"}
for name in expert_names:
    ax.plot(combined["epoch"], combined[f"gate_mean_{name}"], label=name,
            color=colors[name], lw=2)
ax.axvline(GATING_EPOCHS + 0.5, color="grey", ls="--", lw=1)
ax.text(GATING_EPOCHS + 0.5, 0.05, " Stage 2 -> Stage 3", color="grey",
        rotation=90, ha="left", va="bottom", fontsize=9)
ax.set_ylabel("mean gate weight")
ax.set_title("MoE3 Option C — routing-weight evolution across Stage 2 + Stage 3")
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="center right")
ax.grid(True, alpha=0.3)

# Bottom panel: anomaly-score loss (the gate's primary training signal)
ax2 = axes[1]
ax2.plot(combined["epoch"], combined["loss_anom"], color="black", lw=1.5, label="loss_anom = E[anomaly_z^2]")
ax2.axvline(GATING_EPOCHS + 0.5, color="grey", ls="--", lw=1)
ax2.set_xlabel("epoch (combined Stage 2 + Stage 3 axis)")
ax2.set_ylabel("loss_anom")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "MoE3_OptionC_routing_evolution.png")
plt.savefig(plot_path, dpi=130)
plt.close()
print(f"Saved: {plot_path}")


# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 + STEP 3 SUMMARY")
print("=" * 60)
final_gates = stage3_df.iloc[-1][[f"gate_mean_{n}" for n in expert_names]].to_dict()
print(f"  Final mean gate weights (end of Stage 3):")
for n in expert_names:
    print(f"    {n:6s}: {final_gates[f'gate_mean_{n}']:.4f}")
print(f"  Final loss_anom (Stage 3 final epoch): {stage3_df.iloc[-1]['loss_anom']:.4f}")
print(f"  MoE AUC(train-N vs test-N) at end of Stage 3: {moe_auc_shift:.4f}")
print("\nReady for Step 4 (post-training E1-E5 evaluations) when approved.")
