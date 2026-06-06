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
```
The `results/` folder includes both aggregate summaries and subject-level raw result files for each dataset, pipeline, and evaluation regime.

## Reproducibility
The benchmark was implemented in Python using:

- [MOABB](https://moabb.neurotechx.com/docs/index.html)
- [MNE-Python](https://mne.tools/stable/index.html)
- [Braindecode](https://braindecode.org/stable/index.html)
- [PyRiemann](https://github.com/pyRiemann/pyRiemann)
- [scikit-learn](https://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org/)


## Evaluation design
For each dataset, pipelines were evaluated independently under two regimes:

- Within-subject evaluation: each subject is evaluated separately using 5-fold cross-validation.
- Cross-subject evaluation: Leave-one-subject-out evaluation, the model is trained on all subjects except one and tested on the held-out subject.

Quartiles are defined according to within-subject TS+EL ROC-AUC:
- **Q1**: subjects with the lowest within-subject TS+EL scores;
- **Q4**: subjects with the highest within-subject TS+EL scores.

The quartile plots show the mean change: cross-subject ROC-AUC - within-subject ROC-AUC for each pipeline and dataset.

## Notes
The deep-learning models were evaluated using fixed hyperparameters across datasets and regimes. The results should therefore be interpreted as standard-pipeline benchmark behaviour rather than as dataset-specific optimised performance.

