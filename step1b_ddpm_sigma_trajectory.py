"""Step 1b — DDPM sigma_measured trajectory at epochs {100, 200, 300}.

Re-runs DDPM Stage 1 (deterministic, seed=42) and saves state + sigma_measured
at three checkpoints: ep 100, 200, 300. Produces a CSV trajectory of sigma vs
training-epoch parallel to what Step 5 will produce for GAN.

The 300-epoch checkpoint matches the LOCKED checkpoint state. Intermediate
checkpoints are written for completeness; downstream stages use only LOCKED.
"""
import os, random
import numpy as np
import pandas as pd
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

OUT_DIR      = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/extracted"
y_train_path = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_train.csv"
y_test_path  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_test.csv"
WINDOW_SEC = 1.0; T_SEQ = 40; SEED = 42
EXPERT_LR = 1e-3; BATCH_SIZE = 256
SAVE_EPOCHS = (100, 200, 300)
DIFF_T_TOTAL=100; DIFF_T_EVAL=50; DIFF_BETA_LO=1e-4; DIFF_BETA_HI=0.02
DIFF_TIME_DIM=32; DIFF_CHANNELS=(32, 64); SCORE_SIGMA_FLOOR = 0.05


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def get_feature_cols(df):
    return [c for c in df.columns if c not in ["w","y_window","attack_type_window"]]


# --- Data ingestion (verbatim) ---
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
    seqs = [X[s:s+t_seq] for s in range(0, len(df)-t_seq+1, stride)]
    return np.stack(seqs)


train_w = build_global_windows_with_attack(train_pkts, WINDOW_SEC)
feat_cols = get_feature_cols(train_w); D = len(feat_cols)
Xtr_seq = make_sequences(train_w[train_w["y_window"]==0].reset_index(drop=True),
                         t_seq=T_SEQ, stride=1, feature_cols=feat_cols)
scaler = StandardScaler().fit(Xtr_seq.reshape(-1, D))
Xtr_s = scaler.transform(Xtr_seq.reshape(-1, D)).reshape(Xtr_seq.shape).astype(np.float32)
print(f"[data] D={D}, Xtr_s={Xtr_s.shape}")


# --- DDPM architecture (mirror) ---
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


@torch.no_grad()
def expert_scores(expert, X, batch=512):
    expert.eval()
    parts = []
    for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                            batch_size=batch, shuffle=False):
        parts.append(expert.anomaly_score(x.to(device)).cpu().numpy())
    return np.concatenate(parts)


# --- Train + snapshot ---
print("\n" + "=" * 60)
print("Step 1b — DDPM sigma_measured trajectory (deterministic, seed=42)")
print("=" * 60)
set_seed(SEED)
ddpm = DDPM_Expert(D=D, T=T_SEQ, channels=DIFF_CHANNELS).to(device)
opt = optim.Adam(ddpm.parameters(), lr=EXPERT_LR); loss_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(torch.tensor(Xtr_s, dtype=torch.float32)),
                    batch_size=BATCH_SIZE, shuffle=True)

trajectory = []  # rows: (epoch, final_loss_at_epoch, sigma_measured, mu_score, floor_hit)
ddpm.train()
for ep in range(max(SAVE_EPOCHS)):
    losses = []
    for (x,) in loader:
        x = x.to(device)
        pred, noise = ddpm(x)
        loss = loss_fn(pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    if (ep + 1) in SAVE_EPOCHS:
        ddpm.eval()
        s = expert_scores(ddpm, Xtr_s)
        sigma_measured = float(np.std(s)); mu_score = float(np.mean(s))
        floor_hit = sigma_measured < SCORE_SIGMA_FLOOR
        trajectory.append({
            "epoch": ep + 1,
            "final_train_loss_at_epoch": float(np.mean(losses)),
            "sigma_measured": sigma_measured,
            "mu_score": mu_score,
            "floor_hit": bool(floor_hit),
        })
        # save model state
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"MoE3_OptionC_ddpm_ep{ep+1}.pt")
        torch.save({"ddpm_state": ddpm.state_dict(),
                    "epoch": ep + 1,
                    "sigma_measured": sigma_measured,
                    "mu_score": mu_score}, ckpt_path)
        print(f"  ep {ep+1:3d}  loss={trajectory[-1]['final_train_loss_at_epoch']:.6f}  "
              f"mu={mu_score:.4f}  sigma_measured={sigma_measured:.4f}  "
              f"floor_hit={'YES' if floor_hit else 'no'}  "
              f"-> {os.path.basename(ckpt_path)}")
        ddpm.train()

# --- Save trajectory CSV ---
traj_df = pd.DataFrame(trajectory)
traj_csv = os.path.join(RESULTS_DIR, "MoE3_OptionC_ddpm_sigma_trajectory.csv")
traj_df.to_csv(traj_csv, index=False)

print("\nTrajectory:")
print(traj_df.to_string(index=False))
print(f"\nSaved: {traj_csv}")
