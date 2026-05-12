"""Full Stage 1 driver for MoE3-IDS Option C (seed=42).

Runs 100-epoch Stage 1 for each heterogeneous expert (TCN-AE / GAN / DDPM),
then calibrates score normalization with the new sigma floor (0.05).
Reports final losses per expert and final (mu, sigma) values so we can
confirm GAN-D's score variance has stabilised above the floor.

Saves trained experts + score-norm buffers to
checkpoints/MoE3_OptionC_stage1_seed42.pt for use in Stage 2 / 3 later.
"""
import os, random, copy, time
import numpy as np
import pandas as pd
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
SEED       = 42

EXPERT_EPOCHS = 100
EXPERT_LR     = 1e-3
BATCH_SIZE    = 256

DIFF_T_TOTAL  = 100
DIFF_T_EVAL   = 50
DIFF_BETA_LO  = 1e-4
DIFF_BETA_HI  = 0.02
DIFF_TIME_DIM = 32
DIFF_CHANNELS = (32, 64)

GAN_LATENT  = 16
GAN_DISC_LR = 2e-4
GAN_GEN_LR  = 2e-4

N_EXPERTS = 3


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
# Data ingestion (verbatim)
# ============================================================
print("\n[data] Loading cached packet CSVs ...")
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


print("[data] Building windows ...")
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
print(f"[data] D={D}, Xtr_s={Xtr_s.shape}, Xte_s={Xte_s.shape}")


# ============================================================
# Architectures (mirror notebook / smoke test)
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
        self.mid      = ResBlock1D(ch_max, ch_max, time_dim)   # single bottleneck ResBlock
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
    SCORE_SIGMA_FLOOR = 0.05

    def __init__(self, experts, input_dim, num_experts=3, top_k=2, noisy_gating=True):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating  = SparseGatingNetwork(input_dim, num_experts, top_k, noisy_gating)
        self.num_experts = num_experts; self.top_k = top_k
        self.register_buffer("score_mu",            torch.zeros(num_experts))
        self.register_buffer("score_sigma",         torch.ones(num_experts))
        self.register_buffer("score_sigma_measured", torch.ones(num_experts))
        self.calibrated = False

    def calibrate_score_normalization(self, X_train_normal, batch_size=256, expert_names=None):
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
            self.score_mu[i]             = mu_i
            self.score_sigma[i]          = sigma_floored
            self.score_sigma_measured[i] = sigma_measured
            if sigma_measured < self.SCORE_SIGMA_FLOOR:
                tag = expert_names[i] if expert_names else f"expert {i}"
                print(f"  [sigma_floor] {tag}: measured sigma={sigma_measured:.4f} < "
                      f"floor={self.SCORE_SIGMA_FLOOR:.2f}; using floor for z-norm stability")
        self.calibrated = True
        if was_training:
            self.train()


# ============================================================
# Stage 1 trainers
# ============================================================
def pretrain_tcn(expert, X, *, epochs, lr, batch_size, seed):
    set_seed(seed)
    model = expert.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    model.train()
    final_loss = None
    for ep in range(epochs):
        losses = []
        for (x,) in loader:
            x = x.to(device)
            loss = loss_fn(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        final_loss = float(np.mean(losses))
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    [TCN-AE] ep {ep:3d}: loss={final_loss:.6f}")
    return model, final_loss


def pretrain_gan(expert, X, *, epochs, batch_size, seed):
    set_seed(seed)
    model = expert.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt_d = optim.Adam(model.Disc.parameters(), lr=GAN_DISC_LR, betas=(0.5, 0.999))
    opt_g = optim.Adam(model.G.parameters(),    lr=GAN_GEN_LR,  betas=(0.5, 0.999))
    bce = nn.BCELoss()
    model.train()
    final_d, final_g = None, None
    for ep in range(epochs):
        d_losses, g_losses = [], []
        for (x,) in loader:
            x = x.to(device)
            B = x.size(0)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)
            z = torch.randn(B, model.latent, device=device)
            with torch.no_grad():
                x_fake = model.G(z)
            d_real = model.Disc(x); d_fake = model.Disc(x_fake)
            loss_d = bce(d_real, real_lbl) + bce(d_fake, fake_lbl)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            z = torch.randn(B, model.latent, device=device)
            d_fake_for_g = model.Disc(model.G(z))
            loss_g = bce(d_fake_for_g, real_lbl)
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            d_losses.append(loss_d.item()); g_losses.append(loss_g.item())
        final_d, final_g = float(np.mean(d_losses)), float(np.mean(g_losses))
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    [GAN]    ep {ep:3d}: D_loss={final_d:.4f}  G_loss={final_g:.4f}")
    return model, final_d, final_g


def pretrain_ddpm(expert, X, *, epochs, lr, batch_size, seed):
    set_seed(seed)
    model = expert.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    model.train()
    final_loss = None
    for ep in range(epochs):
        losses = []
        for (x,) in loader:
            x = x.to(device)
            pred, true_noise = model(x)
            loss = loss_fn(pred, true_noise)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        final_loss = float(np.mean(losses))
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    [DDPM]   ep {ep:3d}: loss={final_loss:.6f}")
    return model, final_loss


# ============================================================
# Run Stage 1
# ============================================================
print("\n" + "=" * 60)
print(f"STAGE 1 — Option C heterogeneous expert pretraining (seed={SEED})")
print("=" * 60)

set_seed(SEED)
expert_tcn  = TCN_AE_Expert(D, channels=64, kernel_size=3)
expert_gan  = GAN_Expert(latent=GAN_LATENT, T=T_SEQ, D=D, ch=64)
expert_ddpm = DDPM_Expert(D=D, T=T_SEQ, channels=DIFF_CHANNELS)
experts      = [expert_tcn, expert_gan, expert_ddpm]
expert_names = ["TCN-AE", "GAN", "DDPM"]

t_start = time.time()
print(f"\n--- TCN-AE  ({model_size_stats(expert_tcn)['Params']:,} params) ---")
expert_tcn, tcn_final_loss = pretrain_tcn(expert_tcn, Xtr_s,
    epochs=EXPERT_EPOCHS, lr=EXPERT_LR, batch_size=BATCH_SIZE, seed=SEED)
t_tcn = time.time() - t_start

t0 = time.time()
print(f"\n--- GAN     (Disc {model_size_stats(expert_gan.Disc)['Params']:,} infer / "
      f"{model_size_stats(expert_gan)['Params']:,} total trainable) ---")
expert_gan, gan_d_loss, gan_g_loss = pretrain_gan(expert_gan, Xtr_s,
    epochs=EXPERT_EPOCHS, batch_size=BATCH_SIZE, seed=SEED)
t_gan = time.time() - t0

t0 = time.time()
print(f"\n--- DDPM    ({model_size_stats(expert_ddpm)['Params']:,} params) ---")
expert_ddpm, ddpm_final_loss = pretrain_ddpm(expert_ddpm, Xtr_s,
    epochs=EXPERT_EPOCHS, lr=EXPERT_LR, batch_size=BATCH_SIZE, seed=SEED)
t_ddpm = time.time() - t0

print(f"\nStage 1 wall time: TCN-AE {t_tcn:.1f}s  |  GAN {t_gan:.1f}s  |  DDPM {t_ddpm:.1f}s")


# ============================================================
# Calibrate score normalization (sigma floor 0.05)
# ============================================================
print("\n" + "=" * 60)
print("CALIBRATION — score normalization (sigma floor = 0.05)")
print("=" * 60)
moe = MoE_OptionC(experts, input_dim=D, num_experts=N_EXPERTS, top_k=2,
                  noisy_gating=True).to(device)
moe.calibrate_score_normalization(Xtr_s, batch_size=BATCH_SIZE, expert_names=expert_names)


# ============================================================
# Final report
# ============================================================
print("\n" + "=" * 60)
print("STAGE 1 SUMMARY")
print("=" * 60)
print(f"  Final losses (after {EXPERT_EPOCHS} epochs, seed={SEED}):")
print(f"    TCN-AE final MSE-recon loss      : {tcn_final_loss:.6f}")
print(f"    GAN    final D_loss / G_loss     : {gan_d_loss:.4f} / {gan_g_loss:.4f}")
print(f"    DDPM   final noise-prediction MSE: {ddpm_final_loss:.6f}")

print(f"\n  Score-normalization calibration  (sigma floor = {moe.SCORE_SIGMA_FLOOR}):")
mu  = moe.score_mu.cpu().numpy()
sig_used     = moe.score_sigma.cpu().numpy()
sig_measured = moe.score_sigma_measured.cpu().numpy()
print(f"  {'expert':6s}  {'mu':>10s}  {'sigma_measured':>16s}  {'sigma_used':>12s}  floor_hit")
for name, mu_i, sm_i, su_i in zip(expert_names, mu, sig_measured, sig_used):
    hit = "YES" if sm_i < moe.SCORE_SIGMA_FLOOR else "no"
    print(f"  {name:6s}  {mu_i:10.4f}  {sm_i:16.4f}  {su_i:12.4f}  {hit}")

# ============================================================
# Save checkpoint
# ============================================================
ckpt_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionC_stage1_seed{SEED}.pt")
torch.save({
    "tcn_state":   expert_tcn.state_dict(),
    "gan_state":   expert_gan.state_dict(),
    "ddpm_state":  expert_ddpm.state_dict(),
    "score_mu":    moe.score_mu.cpu(),
    "score_sigma": moe.score_sigma.cpu(),
    "score_sigma_measured": moe.score_sigma_measured.cpu(),
    "expert_names": expert_names,
    "config": {
        "D": D, "T_SEQ": T_SEQ, "SEED": SEED,
        "EXPERT_EPOCHS": EXPERT_EPOCHS, "DIFF_CHANNELS": DIFF_CHANNELS,
        "GAN_LATENT": GAN_LATENT, "DIFF_T_EVAL": DIFF_T_EVAL,
        "SCORE_SIGMA_FLOOR": moe.SCORE_SIGMA_FLOOR,
    },
    "stage1_losses": {
        "tcn_recon":  tcn_final_loss,
        "gan_d":      gan_d_loss,
        "gan_g":      gan_g_loss,
        "ddpm_noise": ddpm_final_loss,
    },
}, ckpt_path)
print(f"\nCheckpoint: {ckpt_path}")
print("Stage 1 complete. Ready for Stage 2 / 3 launch when approved.")
