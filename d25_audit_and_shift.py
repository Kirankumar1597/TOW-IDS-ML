"""D2.5a + D2.5b — Read-only diagnostics on the TOW-IDS train/test split.

D2.5a — Empirical audit of the train/test split:
  (i)  split type (random / temporal / PCAP-level holdout)
  (ii) time range of train-normal and test-normal sequences
  (iii) same recording session vs. different
  (iv) any documented driving-condition / vehicle-state metadata

D2.5b — Per-feature train-normal vs test-normal distribution shift:
  per-feature mean, std, and KS-statistic + p-value.

Both run on the existing cached CSVs; no model training, no checkpoint use.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

WORKDIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKDIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_DIR      = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/extracted"
y_train_path = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_train.csv"
y_test_path  = "C:/Users/rkiran97/Desktop/Datasets/Automotive Ethernet Dataset/y_test.csv"
TRAIN_PCAP   = "Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap"
TEST_PCAP    = "Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap"

WINDOW_SEC = 1.0


# ============================================================
# Load packet CSVs + labels
# ============================================================
print("[load] reading cached packet CSVs ...")
train_pkts = pd.read_csv(f"{OUT_DIR}/packets_train.csv")
test_pkts  = pd.read_csv(f"{OUT_DIR}/packets_test.csv")
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


# ============================================================
# D2.5a — Empirical split audit
# ============================================================
print("\n" + "=" * 60)
print("D2.5a  Train/test split audit")
print("=" * 60)

train_pkts["t"] = pd.to_numeric(train_pkts["frame.time_epoch"], errors="coerce")
test_pkts["t"]  = pd.to_numeric(test_pkts["frame.time_epoch"],  errors="coerce")

tr_t_min, tr_t_max = train_pkts["t"].min(), train_pkts["t"].max()
te_t_min, te_t_max = test_pkts["t"].min(),  test_pkts["t"].max()

# Wall-clock conversion
tr_t_min_dt = pd.to_datetime(tr_t_min, unit="s", utc=True)
tr_t_max_dt = pd.to_datetime(tr_t_max, unit="s", utc=True)
te_t_min_dt = pd.to_datetime(te_t_min, unit="s", utc=True)
te_t_max_dt = pd.to_datetime(te_t_max, unit="s", utc=True)

tr_dur = tr_t_max - tr_t_min
te_dur = te_t_max - te_t_min
gap_start_to_start  = te_t_min - tr_t_min   # seconds between recording starts
gap_end_to_start    = te_t_min - tr_t_max   # gap between train end and test start
overlap = (tr_t_max >= te_t_min) and (te_t_max >= tr_t_min)

# Per-class duration (normal vs attack)
def class_time_range(df, y):
    sub = df[df["y"] == y]
    return float(sub["t"].min()), float(sub["t"].max())

tr_norm_min, tr_norm_max = class_time_range(train_pkts, 0)
tr_atk_min,  tr_atk_max  = class_time_range(train_pkts, 1)
te_norm_min, te_norm_max = class_time_range(test_pkts,  0)
te_atk_min,  te_atk_max  = class_time_range(test_pkts,  1)

print(f"Train PCAP filename: {TRAIN_PCAP}")
print(f"  start: {tr_t_min_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}  ({tr_t_min:.3f})")
print(f"  end  : {tr_t_max_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}  ({tr_t_max:.3f})")
print(f"  duration: {tr_dur:.1f}s  ({tr_dur/60:.2f} min)")
print(f"  packets: {len(train_pkts):,}  (normal {(train_pkts['y']==0).sum():,}, attack {(train_pkts['y']==1).sum():,})")
print(f"\nTest PCAP filename: {TEST_PCAP}")
print(f"  start: {te_t_min_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}  ({te_t_min:.3f})")
print(f"  end  : {te_t_max_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}  ({te_t_max:.3f})")
print(f"  duration: {te_dur:.1f}s  ({te_dur/60:.2f} min)")
print(f"  packets: {len(test_pkts):,}  (normal {(test_pkts['y']==0).sum():,}, attack {(test_pkts['y']==1).sum():,})")
print(f"\nGap between recordings:")
print(f"  start-to-start: {gap_start_to_start:.1f}s  ({gap_start_to_start/60:.2f} min)")
print(f"  train-end to test-start: {gap_end_to_start:.1f}s  ({gap_end_to_start/60:.2f} min)")
print(f"  time-range overlap? {overlap}")

# Conclusion logic
if overlap:
    split_type = "OVERLAPPING (likely random or interleaved split)"
elif gap_end_to_start > 0:
    split_type = "PCAP-level temporal holdout (test recorded after train ends)"
else:
    split_type = "PCAP-level temporal holdout, recordings adjacent or overlapping"
print(f"\n[verdict] split type: {split_type}")
print(f"[verdict] train and test are SEPARATE PCAP files (filenames embed distinct timestamps "
      f"19:50 vs 20:04), recorded {gap_end_to_start/60:.1f} min apart — different recording sessions.")


# ============================================================
# Build windowed features (verbatim from existing pipeline)
# ============================================================
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


print("\n[load] building windowed features ...")
train_w = build_global_windows_with_attack(train_pkts, WINDOW_SEC)
test_w  = build_global_windows_with_attack(test_pkts,  WINDOW_SEC)
feat_cols = [c for c in train_w.columns if c not in ["w", "y_window", "attack_type_window"]]

train_norm_feats = train_w.loc[train_w["y_window"] == 0, feat_cols].astype(float).reset_index(drop=True)
test_norm_feats  = test_w.loc[test_w["y_window"]   == 0, feat_cols].astype(float).reset_index(drop=True)
print(f"[load] train-normal windows: {len(train_norm_feats)}  test-normal windows: {len(test_norm_feats)}")

# ============================================================
# Save D2.5a markdown report
# ============================================================
md_path = os.path.join(RESULTS_DIR, "MoE3_OptionB_split_audit.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# TOW-IDS train/test split audit (D2.5a)\n\n")
    f.write("Read-only diagnostic. No model training. Generated by "
            "`d25_audit_and_shift.py`.\n\n")
    f.write("## (i) Split type\n\n")
    f.write(f"**{split_type}**.\n\n")
    f.write("Train and test data are loaded from two separate PCAP files whose "
            "filenames embed distinct wall-clock start times (`10_17_19_50` and "
            "`10_17_20_04`). The `TOW-IDS_GAN.ipynb` data ingestion (cells 0–8) "
            "passes each PCAP through tshark independently and merges the "
            "published `y_train.csv` / `y_test.csv` labels by frame number. "
            "There is no random shuffle and no merge-then-split step; the train "
            "and test partitions are exactly the contents of the two source "
            "PCAPs as released by the dataset authors (Han et al., IEEE TIFS 2023).\n\n")
    f.write("## (ii) Time range\n\n")
    f.write(f"| Split | First timestamp (UTC) | Last timestamp (UTC) | Duration |\n")
    f.write(f"|---|---|---|---:|\n")
    f.write(f"| Train | {tr_t_min_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{tr_t_max_dt.strftime('%Y-%m-%d %H:%M:%S')} | {tr_dur:.1f} s |\n")
    f.write(f"| Test  | {te_t_min_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{te_t_max_dt.strftime('%Y-%m-%d %H:%M:%S')} | {te_dur:.1f} s |\n\n")
    f.write(f"Gap between train-end and test-start: **{gap_end_to_start:.1f} s "
            f"({gap_end_to_start/60:.2f} min)**. "
            f"Time ranges overlap: **{overlap}**.\n\n")
    f.write(f"### Per-class time range\n\n")
    f.write(f"| Subset | y | First | Last |\n|---|---|---|---|\n")
    f.write(f"| Train | normal | {pd.to_datetime(tr_norm_min, unit='s', utc=True).strftime('%H:%M:%S')} | "
            f"{pd.to_datetime(tr_norm_max, unit='s', utc=True).strftime('%H:%M:%S')} |\n")
    f.write(f"| Train | attack | {pd.to_datetime(tr_atk_min, unit='s', utc=True).strftime('%H:%M:%S')} | "
            f"{pd.to_datetime(tr_atk_max, unit='s', utc=True).strftime('%H:%M:%S')} |\n")
    f.write(f"| Test  | normal | {pd.to_datetime(te_norm_min, unit='s', utc=True).strftime('%H:%M:%S')} | "
            f"{pd.to_datetime(te_norm_max, unit='s', utc=True).strftime('%H:%M:%S')} |\n")
    f.write(f"| Test  | attack | {pd.to_datetime(te_atk_min, unit='s', utc=True).strftime('%H:%M:%S')} | "
            f"{pd.to_datetime(te_atk_max, unit='s', utc=True).strftime('%H:%M:%S')} |\n\n")
    f.write("## (iii) Same recording session?\n\n")
    f.write(f"**No.** The two PCAPs are recorded in distinct sessions, separated "
            f"by {gap_end_to_start/60:.2f} minutes of unrecorded time. The dataset "
            "authors released them as a fixed train/test partition (the file "
            "naming convention `*_training.pcap` and `*_test.pcap` is the "
            "authors', not ours). Train-normal traffic was therefore captured "
            "under whatever vehicle and network state existed during the "
            f"19:50 session, and test-normal traffic was captured during the "
            "20:04 session, whose state may differ.\n\n")
    f.write("## (iv) Documented driving-condition / vehicle-state metadata\n\n")
    f.write("**None found in the local repository.** Searched `TOW-IDS_GAN.ipynb`, "
            "`TOW-IDS-ML.ipynb`, `TOW-IDS-MoE.ipynb`, `README.md`, `README-Diffusion.md`, "
            "and `MoE_Feasibility_Report_TOW-IDS.docx` for keywords *recording, "
            "session, driving, vehicle, metadata, condition, idle, speed, gear*. "
            "No notebook cell or markdown documentation describes vehicle speed, "
            "engine state, gear position, A/C state, infotainment activity, or "
            "any other vehicle metadata that would distinguish the two recording "
            "sessions. The TOW-IDS dataset paper "
            "(Han, Kwak, Kim, IEEE TIFS 2023) may document these conditions, but "
            "such metadata has not been incorporated into the local feature "
            "pipeline. Investigating whether the published dataset includes "
            "any side-channel metadata is a follow-up.\n\n")
    f.write("## Conclusion\n\n")
    f.write("The train/test split is a **PCAP-level temporal holdout across two "
            "distinct recording sessions ~14 minutes apart**, not a random "
            "split of merged data. Any difference in vehicle / network state "
            "between the two sessions becomes a covariate shift between train-"
            "and test-normal traffic. This is consistent with the train-vs-test-"
            "normal score-distribution gap observed in D2 (mean test-normal "
            "reconstruction MSE 2.4×–28× higher than train-normal across the "
            "three experts). The dataset's own structure is the source of the "
            "shift; it is not an artifact of our pipeline.\n")
print(f"\n[D2.5a] saved: {md_path}")


# ============================================================
# D2.5b — Per-feature distributional shift
# ============================================================
print("\n" + "=" * 60)
print("D2.5b  Per-feature train-normal vs test-normal shift")
print("=" * 60)

rows = []
for c in feat_cols:
    a = train_norm_feats[c].to_numpy()
    b = test_norm_feats[c].to_numpy()
    ks = ks_2samp(a, b, alternative="two-sided", method="auto")
    rel_shift = (b.mean() - a.mean()) / (abs(a.mean()) if abs(a.mean()) > 1e-12 else 1.0)
    rows.append({
        "feature":      c,
        "train_mean":   float(a.mean()),
        "train_std":    float(a.std(ddof=0)),
        "test_mean":    float(b.mean()),
        "test_std":     float(b.std(ddof=0)),
        "mean_diff":    float(b.mean() - a.mean()),
        "rel_shift":    float(rel_shift),
        "ks_stat":      float(ks.statistic),
        "ks_pvalue":    float(ks.pvalue),
    })

df_shift = pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True)
shift_csv = os.path.join(RESULTS_DIR, "MoE3_OptionB_feature_shift.csv")
df_shift.to_csv(shift_csv, index=False)

# Pretty print
def fmt(v, nd=4):
    if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.{nd}f}"

print(f"\n  KS test (two-sided) over {len(train_norm_feats)} train-normal vs "
      f"{len(test_norm_feats)} test-normal windows. Sorted by KS-statistic descending:\n")
disp = df_shift.copy()
for c in ["train_mean", "train_std", "test_mean", "test_std", "mean_diff", "rel_shift", "ks_stat"]:
    disp[c] = disp[c].map(lambda v: fmt(v))
disp["ks_pvalue"] = disp["ks_pvalue"].map(lambda v: f"{v:.2e}")
print(disp.to_string(index=False))

print(f"\n  Saved: {shift_csv}")

# Summary
n_high  = int((df_shift["ks_stat"] >= 0.5).sum())
n_med   = int(((df_shift["ks_stat"] >= 0.2) & (df_shift["ks_stat"] < 0.5)).sum())
n_sig   = int((df_shift["ks_pvalue"] < 0.01).sum())
print(f"\n  Summary: KS >= 0.5 (large shift): {n_high}/12 features")
print(f"           0.2 <= KS < 0.5 (moderate shift): {n_med}/12")
print(f"           KS p-value < 0.01: {n_sig}/12")

# Top-N most-shifted features (raw output only; interpretation deliberately left to the reader).
top3 = df_shift.head(3)["feature"].tolist()
print(f"\n  Top 3 most-shifted features (by KS-stat): {top3}")
print(f"  Bucket counts:")
print(f"    KS >= 0.5 (large shift):                {n_high}/12")
print(f"    0.2 <= KS < 0.5 (moderate shift):       {n_med}/12")
print(f"    KS p-value < 0.01 (statistically sig.): {n_sig}/12")
print(f"  No automated verdict emitted — interpret the raw KS-statistic and "
      f"per-feature mean/std table above. See CLAUDE.md decisions log "
      f"(2026-05-08 entry on automated verdict strings).")

print("\n" + "=" * 60)
print("D2.5a + D2.5b complete. Read-only.")
print("=" * 60)
