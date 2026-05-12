"""Score-decomposition diagnostic for the trained Option C MoE.

For each of the three sample sets {train_normal, test_normal, test_attack} and
each of the three experts {TCN-AE, GAN, DDPM}, compute:

  (a) raw expert score:                 mean, std
  (b) z-normalized score (sigma_used):  mean, std
  (c) gate-weighted z-score (gate * z): mean, std
  (d/e) per-sample fraction of MoE-score magnitude contributed by each expert:
        frac_i = |gate_i * z_i| / sum_j |gate_j * z_j|, reported as mean/std
        across samples in the sample set.

Run twice:
  - WITH floor:    sigma_used = max(sigma_measured, 0.05)  (training-time)
  - WITHOUT floor: sigma_used = sigma_measured             (clip 1/sigma at 1e3)

Pure forward-pass diagnostic on the Stage-3 trained MoE; no further training.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

WORKDIR        = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(WORKDIR, "checkpoints")
RESULTS_DIR    = os.path.join(WORKDIR, "results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Config
# ============================================================
OUT_DIR      = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/extracted"
y_train_path = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_train.csv"
y_test_path  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_test.csv"

WINDOW_SEC = 1.0; T_SEQ = 40; SEED = 42
N_EXPERTS = 3; TOP_K = 2
DIFF_T_TOTAL=100; DIFF_T_EVAL=50; DIFF_BETA_LO=1e-4; DIFF_BETA_HI=0.02
DIFF_TIME_DIM=32; DIFF_CHANNELS=(32, 64); GAN_LATENT = 16
SCORE_SIGMA_FLOOR = 0.05
INV_SIGMA_CLIP = 1e3   # safety clip on 1/sigma in the no-floor case


def get_feature_cols(df):
    return [c for c in df.columns if c not in ["w","y_window","attack_type_window"]]


# ============================================================
# Data ingestion (verbatim)
# ============================================================
print("\n[data] loading cached packet CSVs ...")
train_pkts = pd.read_csv(f"{OUT_DIR}/packets_train.csv")
test_pkts  = pd.read_csv(f"{OUT_DIR}/packets_test.csv")
y_train = pd.read_csv(y_train_path, header=None,
                      names=["sample_number","normal_or_abnormal","attack_type"])
y_test  = pd.read_csv(y_test_path,  header=None,
                      names=["sample_number","normal_or_abnormal","attack_type"])
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
Xte_normal = Xte_s[yte_seq == 0]
Xte_attack = Xte_s[yte_seq == 1]
print(f"[data] D={D}, train_normal={Xtr_s.shape[0]}, test_normal={Xte_normal.shape[0]}, "
      f"test_attack={Xte_attack.shape[0]}")


# ============================================================
# Architectures (mirror)
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
    def __init__(self, experts, input_dim, num_experts=3, top_k=2, noisy_gating=True):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating  = SparseGatingNetwork(input_dim, num_experts, top_k, noisy_gating)
        self.num_experts = num_experts; self.top_k = top_k
        self.register_buffer("score_mu",    torch.zeros(num_experts))
        self.register_buffer("score_sigma", torch.ones(num_experts))


# ============================================================
# Load Stage 3 trained MoE
# ============================================================
print("\n[load] Stage 3 checkpoint ...")
stage3_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionC_stage3_seed{SEED}.pt")
ckpt3 = torch.load(stage3_path, map_location=device, weights_only=False)
expert_names = list(ckpt3["expert_names"])

tcn  = TCN_AE_Expert(D, channels=64, kernel_size=3).to(device)
gan  = GAN_Expert(latent=GAN_LATENT, T=T_SEQ, D=D, ch=64).to(device)
ddpm = DDPM_Expert(D=D, T=T_SEQ, channels=DIFF_CHANNELS).to(device)
experts = [tcn, gan, ddpm]
moe = MoE_OptionC(experts, input_dim=D, num_experts=N_EXPERTS,
                   top_k=TOP_K, noisy_gating=True).to(device)
moe.load_state_dict(ckpt3["moe_state_dict"])
moe.eval()
mu_used    = moe.score_mu.cpu().numpy().copy()
sigma_used_floor = moe.score_sigma.cpu().numpy().copy()  # values used during training (post-floor)
print(f"  mu               = {np.round(mu_used, 4)}")
print(f"  sigma_used (floor) = {np.round(sigma_used_floor, 4)}")

# Pull the LOCKED measured sigmas — the values the no-floor case will use.
locked_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionC_stage1_seed{SEED}_LOCKED.pt")
locked = torch.load(locked_path, map_location=device, weights_only=False)
sigma_measured_locked = locked["score_sigma_measured"].cpu().numpy().copy()
mu_locked             = locked["score_mu"].cpu().numpy().copy()
print(f"  LOCKED sigma_measured (no floor): {np.round(sigma_measured_locked, 4)}")

# Sanity: mu matches between LOCKED and Stage 3 buffers (should — Stage 3 doesn't update buffers)
assert np.allclose(mu_used, mu_locked, atol=1e-5), "mu drifted between LOCKED and Stage 3"


# ============================================================
# Decomposition helpers
# ============================================================
@torch.no_grad()
def expert_raw_scores_and_gates(moe, X, batch=512):
    """Returns (raw_scores: (B, E), gates: (B, E)) over the full sample set."""
    moe.eval()
    parts_raw, parts_g = [], []
    for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch, shuffle=False):
        x = x.to(device)
        raw = torch.stack([e.anomaly_score(x) for e in moe.experts], dim=1)
        g, _ = moe.gating(x)  # eval mode -> no noise
        parts_raw.append(raw.cpu().numpy()); parts_g.append(g.cpu().numpy())
    return np.concatenate(parts_raw, axis=0), np.concatenate(parts_g, axis=0)


def decompose(raw, gates, mu, sigma, eps=1e-12):
    """Compute z, contributions, |contrib|/sum|contrib|."""
    z = (raw - mu[None, :]) / sigma[None, :]                  # (B, E)
    contrib = gates * z                                        # (B, E)
    abs_contrib = np.abs(contrib)
    total_mag = abs_contrib.sum(axis=1, keepdims=True) + eps
    frac = abs_contrib / total_mag                             # (B, E), rows sum to 1
    moe_score = contrib.sum(axis=1)                            # (B,)
    return z, contrib, frac, moe_score


def per_set_stats(raw, gates, z, contrib, frac):
    """Build a dict of per-expert stats over a sample set."""
    return {
        "raw_mean":           raw.mean(axis=0),
        "raw_std":            raw.std(axis=0),
        "z_mean":             z.mean(axis=0),
        "z_std":              z.std(axis=0),
        "contrib_mean":       contrib.mean(axis=0),
        "contrib_std":        contrib.std(axis=0),
        "frac_mag_mean":      frac.mean(axis=0),
        "frac_mag_std":       frac.std(axis=0),
        "gate_mean":          gates.mean(axis=0),
        "gate_std":           gates.std(axis=0),
    }


# ============================================================
# Compute raw scores + gates once per sample set
# ============================================================
print("\n[forward] computing raw scores + gates on each sample set ...")
sample_sets = {
    "train_normal": Xtr_s,
    "test_normal":  Xte_normal,
    "test_attack":  Xte_attack,
}
raw_per_set, gates_per_set = {}, {}
for name, X in sample_sets.items():
    raw_per_set[name], gates_per_set[name] = expert_raw_scores_and_gates(moe, X)
    print(f"  {name:12s}: raw shape {raw_per_set[name].shape}, "
          f"gates mean {gates_per_set[name].mean(axis=0).round(3).tolist()}")


# ============================================================
# (1) WITH-FLOOR decomposition
# ============================================================
print("\n" + "=" * 60)
print("Decomposition WITH sigma floor (sigma_used = max(sigma_measured, 0.05))")
print("=" * 60)
print(f"  sigma_used = {sigma_used_floor.round(4).tolist()}")
print(f"  1/sigma    = {(1.0 / sigma_used_floor).round(2).tolist()}")
clip_count_floor = int(np.sum(sigma_used_floor < 1.0 / INV_SIGMA_CLIP))
print(f"  1/sigma clip activations (>{INV_SIGMA_CLIP}): {clip_count_floor}")

rows_floor = []
for set_name, X in sample_sets.items():
    raw   = raw_per_set[set_name]
    gates = gates_per_set[set_name]
    z, contrib, frac, moe_score = decompose(raw, gates, mu_used, sigma_used_floor)
    stats = per_set_stats(raw, gates, z, contrib, frac)
    print(f"\n  Sample set: {set_name} (N={raw.shape[0]})")
    print(f"  MoE score: mean={moe_score.mean():+.4f}, std={moe_score.std():.4f}")
    print(f"    {'expert':6s}  {'raw_mean':>10s} {'raw_std':>9s}  "
          f"{'z_mean':>8s} {'z_std':>8s}  {'gate_mean':>9s}  "
          f"{'contrib_mean':>12s} {'contrib_std':>11s}  "
          f"{'frac_mag_mean':>13s} {'frac_mag_std':>12s}")
    for i, name in enumerate(expert_names):
        print(f"    {name:6s}  {stats['raw_mean'][i]:>10.4f} {stats['raw_std'][i]:>9.4f}  "
              f"{stats['z_mean'][i]:>+8.4f} {stats['z_std'][i]:>8.4f}  "
              f"{stats['gate_mean'][i]:>9.4f}  "
              f"{stats['contrib_mean'][i]:>+12.4f} {stats['contrib_std'][i]:>11.4f}  "
              f"{stats['frac_mag_mean'][i]:>13.4f} {stats['frac_mag_std'][i]:>12.4f}")
        rows_floor.append({
            "sample_set": set_name, "expert": name, "n_samples": int(raw.shape[0]),
            "raw_mean":      stats["raw_mean"][i],
            "raw_std":       stats["raw_std"][i],
            "z_mean":        stats["z_mean"][i],
            "z_std":         stats["z_std"][i],
            "gate_mean":     stats["gate_mean"][i],
            "gate_std":      stats["gate_std"][i],
            "contrib_mean":  stats["contrib_mean"][i],
            "contrib_std":   stats["contrib_std"][i],
            "frac_mag_mean": stats["frac_mag_mean"][i],
            "frac_mag_std":  stats["frac_mag_std"][i],
            "sigma_used":    float(sigma_used_floor[i]),
        })

floor_csv = os.path.join(RESULTS_DIR, "MoE3_OptionC_score_decomposition.csv")
pd.DataFrame(rows_floor).to_csv(floor_csv, index=False)
print(f"\nSaved: {floor_csv}")


# ============================================================
# (2) NO-FLOOR decomposition
# ============================================================
print("\n" + "=" * 60)
print("Decomposition WITHOUT sigma floor (sigma_used = sigma_measured)")
print("=" * 60)
sigma_nofloor = sigma_measured_locked.copy()
inv_sigma     = 1.0 / np.maximum(sigma_nofloor, 1e-12)
clip_mask     = inv_sigma > INV_SIGMA_CLIP
n_clipped     = int(clip_mask.sum())
if n_clipped > 0:
    inv_sigma_clipped = np.where(clip_mask, INV_SIGMA_CLIP, inv_sigma)
    sigma_nofloor = 1.0 / inv_sigma_clipped
    print(f"  WARN: clipped {n_clipped} expert(s) at 1/sigma={INV_SIGMA_CLIP}")
else:
    print(f"  1/sigma = {inv_sigma.round(2).tolist()}  -- no clip activations")
print(f"  sigma_used (no-floor) = {sigma_nofloor.round(4).tolist()}")

rows_nofloor = []
for set_name, X in sample_sets.items():
    raw   = raw_per_set[set_name]
    gates = gates_per_set[set_name]
    z, contrib, frac, moe_score = decompose(raw, gates, mu_used, sigma_nofloor)
    stats = per_set_stats(raw, gates, z, contrib, frac)
    print(f"\n  Sample set: {set_name} (N={raw.shape[0]})")
    print(f"  MoE score: mean={moe_score.mean():+.4f}, std={moe_score.std():.4f}")
    print(f"    {'expert':6s}  {'raw_mean':>10s} {'raw_std':>9s}  "
          f"{'z_mean':>8s} {'z_std':>8s}  {'gate_mean':>9s}  "
          f"{'contrib_mean':>12s} {'contrib_std':>11s}  "
          f"{'frac_mag_mean':>13s} {'frac_mag_std':>12s}")
    for i, name in enumerate(expert_names):
        print(f"    {name:6s}  {stats['raw_mean'][i]:>10.4f} {stats['raw_std'][i]:>9.4f}  "
              f"{stats['z_mean'][i]:>+8.4f} {stats['z_std'][i]:>8.4f}  "
              f"{stats['gate_mean'][i]:>9.4f}  "
              f"{stats['contrib_mean'][i]:>+12.4f} {stats['contrib_std'][i]:>11.4f}  "
              f"{stats['frac_mag_mean'][i]:>13.4f} {stats['frac_mag_std'][i]:>12.4f}")
        rows_nofloor.append({
            "sample_set": set_name, "expert": name, "n_samples": int(raw.shape[0]),
            "raw_mean":      stats["raw_mean"][i],
            "raw_std":       stats["raw_std"][i],
            "z_mean":        stats["z_mean"][i],
            "z_std":         stats["z_std"][i],
            "gate_mean":     stats["gate_mean"][i],
            "gate_std":      stats["gate_std"][i],
            "contrib_mean":  stats["contrib_mean"][i],
            "contrib_std":   stats["contrib_std"][i],
            "frac_mag_mean": stats["frac_mag_mean"][i],
            "frac_mag_std":  stats["frac_mag_std"][i],
            "sigma_used":    float(sigma_nofloor[i]),
        })

nofloor_csv = os.path.join(RESULTS_DIR, "MoE3_OptionC_score_decomposition_nofloor.csv")
pd.DataFrame(rows_nofloor).to_csv(nofloor_csv, index=False)
print(f"\nSaved: {nofloor_csv}")


# ============================================================
# AUC(train-N vs test-N) for each scoring scheme — for context
# ============================================================
def auc(neg, pos):
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([neg, pos])
    return float(roc_auc_score(y, s))

print("\n" + "=" * 60)
print("MoE-level AUC(train-N vs test-N) — context")
print("=" * 60)
# WITH floor: same as the trained model; matches the Step 3 finding (0.999)
raw_tr = raw_per_set["train_normal"]; gates_tr = gates_per_set["train_normal"]
raw_tn = raw_per_set["test_normal"];  gates_tn = gates_per_set["test_normal"]
raw_ta = raw_per_set["test_attack"];  gates_ta = gates_per_set["test_attack"]
_, _, _, moe_floor_tr = decompose(raw_tr, gates_tr, mu_used, sigma_used_floor)
_, _, _, moe_floor_tn = decompose(raw_tn, gates_tn, mu_used, sigma_used_floor)
_, _, _, moe_floor_ta = decompose(raw_ta, gates_ta, mu_used, sigma_used_floor)
auc_floor_shift  = auc(moe_floor_tr, moe_floor_tn)
auc_floor_attack = auc(moe_floor_tn, moe_floor_ta)
print(f"  WITH floor    : MoE AUC(train-N vs test-N) = {auc_floor_shift:.4f}   "
      f"AUC(test-N vs test-A) = {auc_floor_attack:.4f}")

_, _, _, moe_nf_tr = decompose(raw_tr, gates_tr, mu_used, sigma_nofloor)
_, _, _, moe_nf_tn = decompose(raw_tn, gates_tn, mu_used, sigma_nofloor)
_, _, _, moe_nf_ta = decompose(raw_ta, gates_ta, mu_used, sigma_nofloor)
auc_nf_shift  = auc(moe_nf_tr, moe_nf_tn)
auc_nf_attack = auc(moe_nf_tn, moe_nf_ta)
print(f"  WITHOUT floor : MoE AUC(train-N vs test-N) = {auc_nf_shift:.4f}   "
      f"AUC(test-N vs test-A) = {auc_nf_attack:.4f}")
print("\nDecompositions complete. No further training. Pausing for review.")
