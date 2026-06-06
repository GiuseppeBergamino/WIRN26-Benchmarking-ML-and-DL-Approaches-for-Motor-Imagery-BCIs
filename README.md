# MI-EEG Decoder Benchmark: Within-Subject vs Cross-Subject Evaluation

This repository contains the code, aggregate results, subject-level results, and figure-generation scripts used for the paper:

**Benchmarking Machine and Deep Learning Approaches for Motor Imagery BCIs**

The benchmark compares within the [MOABB framework](https://moabb.neurotechx.com/docs/index.html) one classical Riemannian machine-learning pipeline and three standard deep-learning architectures for binary left- versus right-hand motor imagery EEG decoding.

## Overview

The study evaluates whether the relative performance of classical ML and DL decoders changes between:
- **within-subject evaluation**, used as a proxy for calibrated subject-specific decoding;
- **leave-one-subject-out cross-subject evaluation**, used as a proxy for subject-independent decoding without target-user data.

The benchmark includes three public MI-EEG datasets:
- `Cho2017`
- `Lee2019`
- `Yang2025`

and four decoding pipelines:
- `TS+EL`: Riemannian tangent-space covariance features with elastic-net logistic regression;
- `EEGNet`
- `ShallowConvNet`
- `DeepConvNet`

All results are reported using ROC-AUC.

## Repository contents
```text
.
├── benchmark/
│   └── benchmark script used to run MOABB evaluations
│
├── results/
│   ├── summary_results.csv
│   ├── Cho2017/
│   │   ├── within/
│   │   └── cross/
│   ├── Lee2019_MI/
│   │   ├── within/
│   │   └── cross/
│   └── Yang2025_2C/
│       ├── within/
│       └── cross/
│
├── figures/
│   ├── scripts used to generate paper figures
│   └── generated figures
│
└── README.md
