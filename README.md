# TOW-IDS-ML: GAN-Based & Advanced ML Intrusion Detection for Automotive Ethernet Traffic

The Jupyter notebook evaluates multiple unsupervised, semi-supervised, and supervised intrusion detection approaches on the Automotive Ethernet-based TOW-IDS dataset, comparing anomaly-detection metrics, inference latency, model size, GPU/CPU utilization, and energy consumption to validate them for edge-deployment scenarios.

## Overview

Modern vehicles rely on Automotive Ethernet for high-bandwidth in-vehicle communication. This project builds and benchmarks multiple anomaly detection models to identify network intrusions (CAN DoS, CAN Replay, Injection, MAC Flooding, PTP sync attacks) from packet-level traffic data.

The pipeline:
1. Extracts packet features from PCAP files using `tshark`
2. Constructs time-windowed statistical features (packet counts, byte sums, inter-arrival times, MAC/IP diversity, etc.)
3. Trains anomaly detection models on **normal traffic only** (unsupervised) or with labeled data (supervised/semi-supervised)
4. Evaluates detection performance across multiple attack types with per-attack breakdown

## Models Implemented

| Model | Type | Architecture | Input Level | Training |
|-------|------|-------------|-------------|----------|
| **GANomaly** | GAN (Encoder-Decoder) | Linear E/D/Discriminator | Window | Unsupervised |
| **GIDS** | GAN (DNN) | Generator + two Discriminators | Window | 1st-D: Supervised, 2nd-D: Unsupervised |
| **MAD-GAN** | GAN (LSTM) | 3-layer LSTM Generator + 1-layer LSTM Discriminator | Sequence | Unsupervised |
| **TadGAN** | WGAN-GP | Bidirectional LSTM Encoder/Generator + Conv1d Critic | Sequence | Unsupervised |
| **INDRA** | Autoencoder | GRU Encoder-Decoder | Sequence | Unsupervised |
| **TENET** | Next-step Predictor | TCN + Self-Attention | Sequence | Unsupervised |
| **LATTE** | Next-step Predictor | LSTM + Self-Attention | Sequence | Unsupervised |

**GIDS** (Seo et al., 2018) uses a two-stage detection approach:
- **1st Discriminator (supervised):** Trained on labeled normal + attack data to detect known attacks
- **2nd Discriminator (unsupervised):** Trained via GAN adversarial process using only normal data to detect unknown/novel attacks

TENET and LATTE are additionally evaluated with second-stage classifiers:
- **TENET + Decision Tree** (supervised calibration)
- **LATTE + One-Class SVM** (semi-supervised)

## Dataset

This project uses the **TOW-IDS Automotive Ethernet Intrusion Detection Dataset**, which contains:
- Training PCAP (normal + attack traffic)
- Test PCAP (normal + attack traffic)
- Per-packet labels (`y_train.csv`, `y_test.csv`) with attack type annotations
- Attack types: CAN DoS (C_D), CAN Replay (C_R), PTP sync (P_I), Injection (F_I), MAC Flooding (M_F)

> **Note:** The dataset is not included in this repository. Update the file paths in the configuration cell to point to your local copy.

## Windowed Features

Each 1-second time window is represented by 13 statistical features:

| Feature | Description |
|---------|-------------|
| `pkt_count` | Number of packets in window |
| `bytes_sum` | Total bytes |
| `pkt_len_mean/std` | Packet length statistics |
| `dt_mean/std` | Inter-arrival time statistics |
| `uniq_src_mac/dst_mac` | Unique MAC address counts |
| `uniq_ip_src/dst` | Unique IP address counts |
| `uniq_eth_type` | Unique Ethernet type count |
| `multicast_ratio` | Fraction of multicast packets |
| `vlan_ratio` | Fraction of VLAN-tagged packets |

## Evaluation

Models are evaluated at multiple granularities:
- **Sequence-level**: Per sliding window of `T_SEQ` consecutive windows
- **Window-level**: Mapped back from sequence scores using mean/max aggregation
- **Packet-level**: Mapped from window predictions to individual packets

### Metrics
- ROC-AUC, PR-AUC
- Precision, Recall, F1-score (at 99th percentile threshold on train-normal scores)
- Per-attack type breakdown (n_attack_seqs, TN/FP/FN/TP per attack)
- Confusion matrices at all levels
- Score distribution histograms

### Deployment Metrics
- Model size (Params, Size in MB)
- Inference latency (Latency in ms)
- Throughput (calls/s)
- Energy consumption (Energy in J/call, measured via NVML)
- GPU Power Avg/Peak (W), GPU Utilization (%), GPU Memory (MB)
- CPU Utilization (%), RSS Peak (MB)

## Ablation Studies

- **Window size**: 0.1s to 2.0s for GANomaly
- **Sequence length** (`T_SEQ`): 10, 20, 40 windows for MAD-GAN
- **Multi-seed** robustness (seeds 0, 1, 2)
- **Scoring iterations** for MAD-GAN (10 to 150) -- latency vs. accuracy tradeoff
- **Score aggregation**: mean vs. max when mapping sequence scores to windows
- **INDRA `reduce_f`**: max vs. mean over feature dimension

## Requirements

- Python 3.8+
- PyTorch (with CUDA recommended)
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- pynvml (for GPU energy measurement)
- psutil (for CPU profiling)
- tshark / Wireshark (for PCAP extraction)

Install dependencies:
```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn pynvml psutil
```

Ensure `tshark` is installed and available on your PATH:
```bash
# Ubuntu/Debian
sudo apt install tshark

# Windows (via Wireshark installer)
# Add Wireshark directory to PATH
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Kirankumar1597/TOW-IDS_ML.git
   cd TOW-IDS_ML
   ```

2. Update the file paths in **Cell 2 (Configuration)** to point to your dataset location:
   ```python
   TRAIN_PCAP = "/path/to/training.pcap"
   TEST_PCAP  = "/path/to/test.pcap"
   y_train_path = "/path/to/y_train.csv"
   y_test_path  = "/path/to/y_test.csv"
   ```

3. Run all cells sequentially in Jupyter Notebook or JupyterLab.

### Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Data Ingestion | PCAP extraction, CSV loading, label merging |
| 2 | Feature Engineering | Window building, sequence construction, scaling utilities |
| 3 | Exploratory Data Analysis | Feature distribution visualization |
| 4 | GANomaly | Window-level GAN anomaly detector + ablations |
| 5 | GIDS | Two-stage GAN IDS (supervised + unsupervised discriminators) |
| 6 | MAD-GAN & TadGAN | Sequence-level GAN models + ablations |
| 7 | Latency-Performance Tradeoff | MAD-GAN scoring iterations sweep |
| 8 | Packet/Window Statistics | Traffic density analysis |
| 9 | Confusion Matrices | Sequence/window/packet level evaluation + granularity analysis |
| 10 | Model Size & Energy | Parameter counts, latency, power measurement |
| 11 | Unified Evaluation | Cross-model comparison at window level |
| 12 | INDRA | GRU autoencoder model |
| 13 | LATTE & TENET | LSTM and TCN attention-based predictors + pipelines |
| 14 | Confusion Matrices | All frameworks: GANomaly, GIDS, MAD-GAN, TadGAN, INDRA, LATTE, TENET |
| 15 | Final Deployment Comparison | Comprehensive profiling of all model variants |

## License

This project is for research and educational purposes.

## Acknowledgments

- TOW-IDS Automotive Ethernet Intrusion Detection Dataset
- Akcay, S., Atapour-Abarghouei, A., & Breckon, T. P. (2018). *GANomaly: Semi-supervised Anomaly Detection via Adversarial Training.* ACCV.
- Seo, E., Song, H. M., & Kim, H. K. (2018). *GIDS: GAN based Intrusion Detection System for In-Vehicle Network.* IEEE PPSC.
- Li, D., Chen, D., Shi, L., Jin, B., Goh, J., & Ng, S.-K. (2019). *MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks.* ICANN.
- Geiger, A., Liu, D., Alnegheimish, S., Cuesta-Infante, A., & Veeramachaneni, K. (2020). *TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks.* IEEE Big Data.
- Kukkala, V. K., Thiruloga, S. V., & Pasricha, S. (2020). *INDRA: Intrusion Detection using Recurrent Autoencoders in Automotive Embedded Systems.* IEEE TCAD.
- Thiruloga, S. V., Kukkala, V. K., & Pasricha, S. (2022). *TENET: Temporal CNN with Attention for Anomaly Detection in Automotive Cyber-Physical Systems.* ASP-DAC.
- Kukkala, V. K., Thiruloga, S. V., & Pasricha, S. (2022). *LATTE: LSTM Self-Attention based Anomaly Detection in Embedded Automotive Platforms.* ACM TECS.
