# MoE3 Consolidated Findings — Input to Paper 1 draft and advisor conversation

**Generated 2026-05-08.** Closes the Option B + Option C experimental sequence. Source documents: `MoE3_design_spec.md` (canonical spec, including the four-findings summary at the top of Option C and the Open Questions section), `CLAUDE.md` Decisions Log entries dated 2026-05-07 and 2026-05-08.

This file is the consolidated input to (a) the Paper 1 draft and (b) the advisor conversation. It is not a paper outline. It states the four findings as numbered claims, points to the figures/CSVs that support each, and lists what is still missing before Paper 1 is submittable.

---

## Finding 1 — Train/test covariate shift dominates the operating point

**Claim.** TOW-IDS as released is a PCAP-level temporal holdout across two distinct recording sessions ~3 minutes apart, not a random split. Per-feature distributional shifts between train-normal and test-normal traffic are mild in magnitude (3–6% relative shift in the most-shifted features) but statistically very significant (KS p < 10⁻¹⁷ on three of twelve features) and broadly distributed across volume / timing / multicast features. Reconstruction-AE-style expert scoring functions amplify these mild input shifts into 13× to 28× score-distribution gaps between train-normal and test-normal, which alone is sufficient to break p99-train-thresholded operating points.

**Single-number summary.** AUC(train-N vs test-N) of standalone experts: GAT-AE 0.621 (least shift seen), GRU-AE 0.940, TCN-AE 0.996, MoE 4-expert 0.995, MoE 3-expert 0.999. Values near 1.0 indicate the expert treats test-normal as a different class from train-normal.

**Supporting artifacts.**
- [MoE3_OptionB_split_audit.md](MoE3_OptionB_split_audit.md) — empirical split audit (D2.5a). Time ranges, session gap, no documented driving / vehicle metadata.
- [MoE3_OptionB_feature_shift.csv](MoE3_OptionB_feature_shift.csv) — per-feature KS test, mean/std for train-normal vs test-normal (D2.5b).
- [MoE3_OptionB_normal_vs_normal_auc.csv](MoE3_OptionB_normal_vs_normal_auc.csv) — AUC(train-N vs test-N) per model on Option B (D2.5c).
- [MoE3_OptionB_score_distributions.png](MoE3_OptionB_score_distributions.png) — Paper 1 candidate Figure 1: 4-expert score distributions in 2×2 panel format.
- [MoE3_OptionB_validation_threshold.csv](MoE3_OptionB_validation_threshold.csv) — D2 result that validation-derived thresholding does *not* fix the shift (because validation is drawn from the same training session).

---

## Finding 2 — Capacity-driven shift fragility in reconstruction MoE

**Claim.** Train-fit tightness scales monotonically with expert capacity, and tighter train-fit produces stricter shift fragility. The smallest expert is the most deployment-robust because its wider train-score distribution accidentally bridges the train-vs-test-normal score gap. The expert with the lowest train-normal MSE is *not* the expert with the best deployment F1.

**Numbers (Option B, seed=42).**

| Expert | Params | Stage 1 final MSE | Standalone p99 F1 | Standalone ROC-AUC | AUC(train-N vs test-N) |
|---|---:|---:|---:|---:|---:|
| GAT-AE | 16K | 0.821 | 0.977 | 0.976 | **0.621** |
| GRU-AE | 65K | 0.118 | 0.977 | 0.973 | 0.940 |
| TCN-AE | 102K | 0.078 | **0.905** | 0.970 | **0.996** |

TCN-AE achieves the lowest train-MSE and the highest gate weight in the 3-expert MoE (87% mean) but flags every test sequence — including all 62 normals — as anomalous at p99 (FP=62/62, precision 0.83). GAT-AE's higher train-MSE acts as implicit regularization.

**Supporting artifacts.**
- [MoE3_OptionB_diagnostic.csv](MoE3_OptionB_diagnostic.csv) — Stage 1 convergence + standalone evaluation per expert.
- [MoE3_OptionB_loss_curves.csv](MoE3_OptionB_loss_curves.csv) — full per-epoch Stage 1 loss curves.
- [MoE3_OptionB_threshold_sweep.csv](MoE3_OptionB_threshold_sweep.csv) — D1 percentile-threshold sensitivity, ranges {p90, p92, p95, p97, p99, p99.5, p99.9}.
- [MoE3_OptionB_threshold_sweep.png](MoE3_OptionB_threshold_sweep.png) — PR curves with threshold markers, showing TCN-AE has no usable operating point in [p90, p99.5].
- [MoE3_OptionB_results.csv](MoE3_OptionB_results.csv), [MoE3_OptionB_per_attack.csv](MoE3_OptionB_per_attack.csv), [MoE3_OptionB_routing.csv](MoE3_OptionB_routing.csv) — headline metrics, per-attack table, routing weights.

---

## Finding 3 — Paradigm-independent score-distribution collapse

**Claim.** Reconstruction (TCN-AE), adversarial discriminator (GAN-D), and diffusion noise-prediction (DDPM) all exhibit train-normal score-distribution collapse. Three distinct mechanisms converge on the same operational consequence: σ_measured shrinks during training and percentile-thresholded operating-point evaluation breaks.

**Three mechanisms.**
- **Capacity-driven overfitting (TCN-AE).** As capacity grows, train-MSE drops and σ of the train-normal MSE distribution shrinks (already documented in Finding 2).
- **Nash-equilibrium degeneracy (GAN-D).** After 100 epochs, σ_measured of `1 − D(x)` on train-normals = 0.0107. The discriminator outputs ≈ 0.518 on essentially every train-normal sequence, the equilibrium where D cannot distinguish G's outputs from real normals.
- **Deterministic-noise tightening (DDPM).** With fixed-seed inference noise at fixed t=50, σ_measured of the noise-prediction MSE distribution shrinks monotonically: 0.0345 → 0.0297 → 0.0275 at epochs 100 → 200 → 300. Loss continues to decrease (0.32 → 0.23) but score variance shrinks faster than mean.

**Generalisation.** Paper 1's contribution extends from "capacity-driven shift fragility in reconstruction MoE" to "paradigm-independent operating-point fragility in small-data anomaly detection."

**Supporting artifacts.**
- [MoE3_OptionC_ddpm_extension.csv](MoE3_OptionC_ddpm_extension.csv) — full 300-epoch DDPM loss curve (per-epoch).
- [MoE3_OptionC_ddpm_extension.png](MoE3_OptionC_ddpm_extension.png) — loss-curve plot with the 100-epoch boundary marked.
- [MoE3_OptionC_ddpm_sigma_trajectory.csv](MoE3_OptionC_ddpm_sigma_trajectory.csv) — σ_measured at epochs {100, 200, 300} for DDPM (parallel to the planned-but-not-run GAN trajectory).
- [MoE3_OptionC_stage3_final.csv](MoE3_OptionC_stage3_final.csv) — final-epoch standalone AUC(train-N vs test-N) per Option C expert: TCN-AE 0.996, GAN 0.834, DDPM 1.000.

---

## Finding 4 — Score-normalization decoupling

**Claim.** z-normalization with σ floor cannot compensate for expert-dependent shift-amplification factors. The σ floor stabilises numerics but decouples gate weights from actual MoE score-magnitude contributions. Removing the floor restores the coupling but trades AUC(test-N vs test-A). No constant σ_used simultaneously equalises cross-paradigm shift behavior and preserves detection signal. The original GenMoE-IDS design (heterogeneous experts + z-normalization) is structurally incoherent for shifted-test small-data deployment and requires redesign before becoming viable.

**Decomposition numbers (test-normal sample set, Stage-3 trained MoE).**

| Expert | Gate (mean weight) | WITH floor: frac of \|MoE\| magnitude | WITHOUT floor: frac of \|MoE\| magnitude |
|---|---:|---:|---:|
| TCN-AE | 0.243 | **0.690** | 0.367 |
| GAN | 0.758 | **0.310** | 0.633 |
| DDPM | 0.000 | 0.000 | 0.000 |

**MoE-level AUCs — the floor's trade.**
- WITH floor:    AUC(train-N vs test-N) = 0.999, AUC(test-N vs test-A) = 0.970.
- WITHOUT floor: AUC(train-N vs test-N) = 0.984, AUC(test-N vs test-A) = 0.876.

Removing the floor reduces shift fragility by 0.015 but reduces detection AUC by 0.094. The floor's job is therefore not "numerical safety" — it is a hidden architectural choice that reroutes the MoE's effective behavior.

**Mechanism.** TCN-AE raw scores blow up by ~600,000× from train-normal mean (0.084) to test-attack mean (49,650). Divided by σ_used = 0.0712, TCN-AE's z-scores reach mean +697,106 on test-attack and dominate the MoE score regardless of gate weight. The gate has correctly minimised its training objective `mean(anomaly_z²)` on train-normals — by routing to GAN, whose floored σ keeps GAN's z-scores small — but the gate's choice does not control the MoE's deployment score magnitude.

**Supporting artifacts.**
- [MoE3_OptionC_score_decomposition.csv](MoE3_OptionC_score_decomposition.csv) — per-expert raw / z / contribution stats with σ floor active, on three sample sets.
- [MoE3_OptionC_score_decomposition_nofloor.csv](MoE3_OptionC_score_decomposition_nofloor.csv) — same decomposition with σ_measured used directly (no floor, no clip activations).
- [MoE3_OptionC_stage2_traces.csv](MoE3_OptionC_stage2_traces.csv) — per-epoch gating evolution and per-expert z-statistics across 50 Stage 2 epochs.
- [MoE3_OptionC_stage3_traces.csv](MoE3_OptionC_stage3_traces.csv) — same trace logging across 30 Stage 3 epochs.
- [MoE3_OptionC_routing_evolution.png](MoE3_OptionC_routing_evolution.png) — combined Stage 2 + Stage 3 routing-weight plot.

---

## What's still missing for Paper 1

The four findings above are the proposed contributions. Before Paper 1 is submittable, the following gaps need to be closed:

1. **Cross-dataset replication.** All findings to date come from the single TOW-IDS train/test split. Replication on at least one of ROAD or Car-Hacking is the main missing item — both have published normal/attack splits and would test whether the four findings are TOW-IDS artifacts or generalize. ROAD is the stronger candidate because its split structure differs from TOW-IDS (multiple recording sessions, longer duration). **This is the headline gap.**

2. **Multi-seed runs.** Every reported number above is from a single seed (42). Variance across seeds is unknown. Paper 1's reporting convention should be `mean ± std over 5 seeds` (per `MoE3_design_spec.md` § Reporting Conventions). Re-running Stages 1 / 2 / 3 across seeds {0, 1, 2, 3, 4} is cheap (~5 minutes total) but has not been done.

3. **GAN-duration σ trajectory.** Step 5 of the Option C protocol planned a GAN σ_measured trajectory at epochs {50, 100, 200} parallel to the DDPM trajectory in Finding 3. Not run (the protocol was halted after the score-decomposition finding). Without it, the "paradigm-independent collapse" claim has DDPM evidence at three points but only a single-point GAN snapshot. Adding three GAN trajectory points would close the symmetry. ~5 minutes of re-training.

4. **Structured σ-floor ablation (E5).** Original protocol step E5 — re-evaluate the trained MoE with `σ_used = σ_measured` (no floor) on the standard metric panel — was halted in favor of the decomposition diagnostic. The decomposition gives the mechanistic story; E5 would give the headline number table for the paper (ROC-AUC, PR-AUC, F1, AUC(train-N vs test-N) at p99 train + validation thresholds, with vs without floor). Not strictly required but completes the standard ablation table.

5. **Statistical significance.** AUC differences of 0.01–0.05 are reported across configurations. Whether these are within seed variance is unverified. A paired bootstrap on AUC differences would resolve this and is a 1-day add.

6. **Figure 1 polish.** [MoE3_OptionB_score_distributions.png](MoE3_OptionB_score_distributions.png) is the candidate Paper 1 Figure 1. The auto-rendered version is functional; for submission it needs typography, axis-label cleanup, and a caption. The Option C decomposition table also wants a sister figure showing gate-weight-vs-contribution-fraction directly — currently only in the CSVs.

## What to do with Option C and Paper 2

Option C is closed (`MoE3_design_spec.md` § "Open questions for follow-up work" lists three Paper 2 directions as candidates without recommendation). The Paper 2 question — whether a viable heterogeneous-expert MoE for automotive AD exists at all — is open. The original GenMoE-IDS proposal (heterogeneous experts + z-normalization) is no longer the right framing. Paper 2 needs a redesign conversation with the advisor before any prototype work.

## Pointers to source documents

- `MoE3_design_spec.md` — canonical spec, with the four-findings summary embedded under § Option C and the Open Questions section appended. Option C's original content is preserved as the experimental record.
- `CLAUDE.md` Decisions Log — dated entries 2026-05-07 (Option B / GAT-AE / dual accounting) and 2026-05-08 (paradigm-independent collapse, score-normalization decoupling, DDPM finalisation, dual accounting).
- `MoE3-IDS-OptionB.ipynb`, `MoE3-IDS-OptionC.ipynb` — canonical notebooks; the standalone driver scripts `full_train_moe3.py`, `full_stage1_optionc.py`, `step1_extend_ddpm.py`, `step1b_ddpm_sigma_trajectory.py`, `step2_3_stages_optionc.py`, `score_decomposition_optionc.py`, `threshold_diagnostics.py`, `d25_audit_and_shift.py`, `d25c_normal_vs_normal_auc.py`, `diagnostic_moe3.py` produced the artifacts referenced above.
- `checkpoints/` — `MoE3_OptionB_seed42.pt`, `MoE3_OptionC_stage1_seed42.pt`, `MoE3_OptionC_stage1_seed42_LOCKED.pt`, `MoE3_OptionC_stage3_seed42.pt`, plus the DDPM intermediate checkpoints at epochs {100, 200, 300}.
