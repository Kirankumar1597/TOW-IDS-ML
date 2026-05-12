"""Step 1 of the Option C 5-step protocol.

Extend DDPM Stage 1 to 300 total epochs (seed=42, deterministic — first 100
epochs reproduce the prior 100-epoch run). Re-calibrate score normalization
on the converged checkpoint. Save as MoE3_OptionC_stage1_seed42_LOCKED.pt
(immutable for downstream stages).

Reports:
  - per-epoch loss curve (full 300 epochs) saved to results/MoE3_OptionC_ddpm_extension.csv
  - sigma_measured at epoch 100 and at epoch 300 (DDPM)
  - whether sigma rises naturally above the 0.05 floor
  - final loss and final calibration buffers for the LOCKED checkpoint
"""
import os, random, time, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

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

WINDOW_SEC = 1.0
T_SEQ      = 40
SEED       = 42
EXPERT_LR  = 1e-3
BATCH_SIZE = 256
DDPM_TOTAL_EPOCHS = 300
DDPM_PRIOR_EPOCHS = 100   # snapshot at this epoch for "before" sigma

DIFF_T_TOTAL  = 100
DIFF_T_EVAL   = 50
DIFF_BETA_LO  = 1e-4
DIFF_BETA_HI  = 0.02
DIFF_TIME_DIM = 32
DIFF_CHANNELS = (32, 64)

GAN_LATENT  = 16
SCORE_SIGMA_FLOOR = 0.05


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_feature_cols(df):
    return [c for c in df.columns if c not in ["w", "y_window", "attack_type_window"]]


def model_size_stats(m):
    return {"Params": int(sum(p.numel() for p in m.parameters()))}


# ============================================================
# Data ingestion
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
            at_col = c; break
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


train_w = build_global_windows_with_attack(train_pkts, WINDOW_SEC)
test_w  = build_global_windows_with_attack(test_pkts,  WINDOW_SEC)
feat_cols = get_feature_cols(train_w)
D = len(feat_cols)
Xtr_seq, _ = make_sequences(
    train_w[train_w["y_window"] == 0].reset_index(drop=True),
    t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
Xte_seq, yte_seq = make_sequences(test_w, t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
scaler = StandardScaler().fit(Xtr_seq.reshape(-1, D))
Xtr_s = scaler.transform(Xtr_seq.reshape(-1, D)).reshape(Xtr_seq.shape).astype(np.float32)
Xte_s = scaler.transform(Xte_seq.reshape(-1, D)).reshape(Xte_seq.shape).astype(np.float32)
print(f"[data] D={D}, Xtr_s={Xtr_s.shape}")


# ============================================================
# Architectures (must mirror full_stage1_optionc.py exactly)
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
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=(1, 2))


class TS_Generator(nn.Module):
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
    def forward(self, x_seq):
        return self.Disc(x_seq).squeeze(-1)
    def anomaly_score(self, x_seq):
        d = self.Disc(x_seq).squeeze(-1)
        return 1.0 - d


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim=DIFF_TIME_DIM):
        super().__init__()
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
    def forward(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(-np.log(10000.0) *
                           torch.arange(half, device=t.device).float() / max(half, 1))
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


class ResBlock1D(nn.Module):
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
    def __init__(self, D=12, channels=DIFF_CHANNELS, time_dim=DIFF_TIME_DIM):
        super().__init__()
        ch_mid, ch_max = channels
        self.time_emb = TimeEmbedding(time_dim)
        self.in_proj  = nn.Conv1d(D, ch_mid, 3, padding=1)
        self.down1    = ResBlock1D(ch_mid, ch_mid, time_dim)
        self.down2    = ResBlock1D(ch_mid, ch_max, time_dim)
        self.mid      = ResBlock1D(ch_max, ch_max, time_dim)
        self.up1      = ResBlock1D(ch_max + ch_max, ch_mid, time_dim)
        self.up2      = ResBlock1D(ch_mid + ch_mid, ch_mid, time_dim)
        self.out_proj = nn.Conv1d(ch_mid, D, 3, padding=1)
    def forward(self, x_seq, t):
        x = x_seq.transpose(1, 2)
        t_emb = self.time_emb(t)
        h0 = self.in_proj(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(h1, t_emb)
        h  = self.mid(h2, t_emb)
        h  = self.up1(torch.cat([h, h2], dim=1), t_emb)
        h  = self.up2(torch.cat([h, h1], dim=1), t_emb)
        return self.out_proj(h).transpose(1, 2)


class DDPM_Expert(nn.Module):
    def __init__(self, D=12, T=T_SEQ, channels=DIFF_CHANNELS,
                 t_total=DIFF_T_TOTAL, t_eval=DIFF_T_EVAL,
                 beta_lo=DIFF_BETA_LO, beta_hi=DIFF_BETA_HI,
                 time_dim=DIFF_TIME_DIM, eval_seed=42):
        super().__init__()
        self.D, self.T = D, T
        self.t_total = t_total
        self.t_eval  = t_eval
        self.eval_seed = eval_seed
        self.unet = Diffusion1DUNet(D=D, channels=channels, time_dim=time_dim)
        betas      = torch.linspace(beta_lo, beta_hi, t_total)
        alphas     = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)
    def add_noise(self, x_0, t, noise=None):
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
        B = x_seq.size(0)
        t = torch.randint(0, self.t_total, (B,), device=x_seq.device)
        x_t, noise = self.add_noise(x_seq, t)
        pred = self.unet(x_t, t)
        return pred, noise
    def anomaly_score(self, x_seq):
        B = x_seq.size(0)
        gen = torch.Generator(device=x_seq.device).manual_seed(self.eval_seed)
        noise = torch.empty_like(x_seq).normal_(generator=gen)
        t = torch.full((B,), self.t_eval, device=x_seq.device, dtype=torch.long)
        x_t, true_noise = self.add_noise(x_seq, t, noise=noise)
        pred = self.unet(x_t, t)
        return torch.mean((pred - true_noise) ** 2, dim=(1, 2))


# ============================================================
# Score helpers
# ============================================================
@torch.no_grad()
def expert_scores(expert, X, batch=512):
    expert.eval()
    parts = []
    for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch, shuffle=False):
        parts.append(expert.anomaly_score(x.to(device)).cpu().numpy())
    return np.concatenate(parts)


# ============================================================
# Step 1: extend DDPM Stage 1 to 300 epochs (deterministic, seed=42)
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 1 — Extend DDPM Stage 1 to {DDPM_TOTAL_EPOCHS} total epochs (seed={SEED})")
print("=" * 60)

set_seed(SEED)
ddpm = DDPM_Expert(D=D, T=T_SEQ, channels=DIFF_CHANNELS).to(device)
print(f"[ddpm] params: {model_size_stats(ddpm)['Params']:,}")
opt = optim.Adam(ddpm.parameters(), lr=EXPERT_LR)
loss_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32)),
                    batch_size=BATCH_SIZE, shuffle=True)

per_epoch_loss = []
ddpm_state_at_100 = None
sigma_at_100 = None

t_start = time.time()
ddpm.train()
for ep in range(DDPM_TOTAL_EPOCHS):
    losses = []
    for (x,) in loader:
        x = x.to(device)
        pred, true_noise = ddpm(x)
        loss = loss_fn(pred, true_noise)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    per_epoch_loss.append(float(np.mean(losses)))
    if ep % 25 == 0 or ep == DDPM_TOTAL_EPOCHS - 1:
        print(f"  ep {ep:3d}: loss={per_epoch_loss[-1]:.6f}")
    # Snapshot at the prior 100-epoch boundary
    if ep == DDPM_PRIOR_EPOCHS - 1:
        ddpm_state_at_100 = {k: v.detach().cpu().clone() for k, v in ddpm.state_dict().items()}
        # measure sigma at this checkpoint without disturbing training state
        ddpm.eval()
        s100 = expert_scores(ddpm, Xtr_s)
        sigma_at_100 = float(np.std(s100))
        ddpm.train()
        print(f"  [snapshot ep {ep+1}] sigma_measured = {sigma_at_100:.4f}  "
              f"(prior 100-epoch run reported 0.0352 — should match)")

t_elapsed = time.time() - t_start
print(f"\nDDPM extension wall time: {t_elapsed:.1f}s")
print(f"Final loss (ep {DDPM_TOTAL_EPOCHS}): {per_epoch_loss[-1]:.6f}")

# Save per-epoch loss curve CSV
curve_csv = os.path.join(RESULTS_DIR, "MoE3_OptionC_ddpm_extension.csv")
pd.DataFrame({"epoch": list(range(1, DDPM_TOTAL_EPOCHS + 1)),
              "ddpm_loss": per_epoch_loss}).to_csv(curve_csv, index=False)
print(f"Per-epoch loss curve: {curve_csv}")

# Plot the loss curve with the 100-epoch boundary
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, DDPM_TOTAL_EPOCHS + 1), per_epoch_loss, lw=1.2, color="C0")
ax.axvline(DDPM_PRIOR_EPOCHS, color="grey", ls="--", lw=1, label="prior 100-ep boundary")
ax.set_xlabel("epoch"); ax.set_ylabel("noise-prediction MSE")
ax.set_title("DDPM Stage 1 extended to 300 epochs (seed=42)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "MoE3_OptionC_ddpm_extension.png")
plt.savefig(plot_path, dpi=130)
plt.close()
print(f"Loss-curve plot: {plot_path}")

# ============================================================
# Re-calibrate sigma on the converged 300-epoch DDPM (DDPM only, direct)
# ============================================================
print("\n[sigma] direct DDPM sigma_measured at convergence:")
ddpm.eval()
s300 = expert_scores(ddpm, Xtr_s)
sigma_at_300 = float(np.std(s300))
mu_at_300    = float(np.mean(s300))
print(f"  sigma_measured at ep 100: {sigma_at_100:.4f}")
print(f"  sigma_measured at ep 300: {sigma_at_300:.4f}")
print(f"  delta:                    {sigma_at_300 - sigma_at_100:+.4f}")
above_floor = sigma_at_300 >= SCORE_SIGMA_FLOOR
print(f"  rises above 0.05 floor naturally? {'YES' if above_floor else 'NO'}")

# ============================================================
# Build LOCKED checkpoint: load TCN-AE and GAN from prior checkpoint;
# use the converged DDPM. Run full MoE_OptionC calibration to get all 3 sigmas.
# ============================================================
prior_ckpt_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionC_stage1_seed{SEED}.pt")
print(f"\n[locked] loading TCN-AE + GAN from prior checkpoint: {prior_ckpt_path}")
prior = torch.load(prior_ckpt_path, map_location=device, weights_only=False)
tcn = TCN_AE_Expert(D, channels=64, kernel_size=3).to(device)
gan = GAN_Expert(latent=GAN_LATENT, T=T_SEQ, D=D, ch=64).to(device)
tcn.load_state_dict(prior["tcn_state"])
gan.load_state_dict(prior["gan_state"])
# Sanity: prior calibration values from the locked TCN/GAN
prior_mu    = prior["score_mu"].cpu().numpy()
prior_sigma = prior["score_sigma_measured"].cpu().numpy()
print(f"  prior TCN-AE: mu={prior_mu[0]:.4f}, sigma_measured={prior_sigma[0]:.4f}")
print(f"  prior GAN   : mu={prior_mu[1]:.4f}, sigma_measured={prior_sigma[1]:.4f}")
print(f"  prior DDPM  : mu={prior_mu[2]:.4f}, sigma_measured={prior_sigma[2]:.4f}  "
      f"(should match ep 100 snapshot {sigma_at_100:.4f})")

# Compute mu/sigma on each expert directly
print("\n[locked] full calibration on the LOCKED triple (TCN, GAN, DDPM_300):")
expert_names = ["TCN-AE", "GAN", "DDPM"]
experts_locked = [tcn, gan, ddpm]
mu_locked    = np.zeros(3)
sigma_meas_locked = np.zeros(3)
sigma_used_locked = np.zeros(3)
for i, (name, exp) in enumerate(zip(expert_names, experts_locked)):
    s = expert_scores(exp, Xtr_s)
    mu_i = float(np.mean(s))
    sigma_i = float(np.std(s))
    sigma_used = max(sigma_i, SCORE_SIGMA_FLOOR)
    mu_locked[i] = mu_i
    sigma_meas_locked[i] = sigma_i
    sigma_used_locked[i] = sigma_used
    floor_hit = "YES" if sigma_i < SCORE_SIGMA_FLOOR else "no"
    print(f"  {name:6s}  mu={mu_i:8.4f}  sigma_measured={sigma_i:.4f}  "
          f"sigma_used={sigma_used:.4f}  floor_hit={floor_hit}")

# ============================================================
# Save LOCKED checkpoint
# ============================================================
locked_ckpt_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionC_stage1_seed{SEED}_LOCKED.pt")
torch.save({
    "tcn_state":  tcn.state_dict(),
    "gan_state":  gan.state_dict(),
    "ddpm_state": ddpm.state_dict(),
    "score_mu":             torch.tensor(mu_locked, dtype=torch.float32),
    "score_sigma":          torch.tensor(sigma_used_locked, dtype=torch.float32),
    "score_sigma_measured": torch.tensor(sigma_meas_locked, dtype=torch.float32),
    "expert_names": expert_names,
    "config": {
        "D": D, "T_SEQ": T_SEQ, "SEED": SEED,
        "TCN_EPOCHS":  100,
        "GAN_EPOCHS":  100,
        "DDPM_EPOCHS": DDPM_TOTAL_EPOCHS,
        "DIFF_CHANNELS": DIFF_CHANNELS,
        "GAN_LATENT": GAN_LATENT, "DIFF_T_EVAL": DIFF_T_EVAL,
        "SCORE_SIGMA_FLOOR": SCORE_SIGMA_FLOOR,
    },
    "stage1_losses": {
        **prior.get("stage1_losses", {}),
        "ddpm_noise_300ep": per_epoch_loss[-1],
    },
    "ddpm_loss_curve": per_epoch_loss,
    "ddpm_sigma_at_100": sigma_at_100,
    "ddpm_sigma_at_300": sigma_at_300,
    "locked": True,
}, locked_ckpt_path)
print(f"\n[LOCKED] checkpoint: {locked_ckpt_path}")
print("This file is treated as immutable for downstream Stages 2 / 3.")

print("\n" + "=" * 60)
print("STEP 1 SUMMARY")
print("=" * 60)
print(f"  DDPM extended: 100 -> {DDPM_TOTAL_EPOCHS} epochs")
print(f"  Final loss (ep 300):     {per_epoch_loss[-1]:.6f}")
print(f"  Loss at ep 100:          {per_epoch_loss[DDPM_PRIOR_EPOCHS - 1]:.6f}")
print(f"  Loss reduction:          {per_epoch_loss[DDPM_PRIOR_EPOCHS - 1] - per_epoch_loss[-1]:+.6f}")
print(f"  sigma_measured ep 100:   {sigma_at_100:.4f}  (floor 0.05, hit: YES)")
print(f"  sigma_measured ep 300:   {sigma_at_300:.4f}  "
      f"(floor 0.05, hit: {'YES' if sigma_at_300 < SCORE_SIGMA_FLOOR else 'no'})")
print(f"  rises above floor naturally? {'YES' if above_floor else 'NO'}")
