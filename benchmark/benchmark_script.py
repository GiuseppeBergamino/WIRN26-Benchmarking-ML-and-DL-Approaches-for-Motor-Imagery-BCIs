# run_mi_benchmark.py

import time
import warnings
from datetime import datetime
from pathlib import Path

import mne
import moabb
import pandas as pd
import torch

from moabb.datasets import Cho2017
from moabb.datasets import Lee2019_MI as MOABBLee2019_MI
from moabb.datasets import Yang2025 as MOABBYang2025

from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation, CrossSubjectEvaluation
from moabb.utils import setup_seed

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from braindecode import EEGClassifier
from braindecode.models import EEGNet, ShallowFBCSPNet, Deep4Net

from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ===================== LOGGING ==========================================

moabb.set_log_level("ERROR")
mne.set_log_level("CRITICAL")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*Pipeline instance is not fitted yet.*")


# ======================= USER SETTINGS =================================

# Datasets: "Cho2017", "Lee2019_MI", "Yang2025_2C"
DATASET_NAME = "Yang2025_2C"

EVAL_KIND = "cross"  # "within" or "cross"

# Pipelines: "TS+EL", "EEGNet_8_2", "ShallowConvNet", "DeepConvNet"
PIPELINE_NAME = "DeepConvNet"

# None keeps the original dataset sampling rate.
# EEGNet: 128 Hz
# ShallowConvNet and DeepConvNet: 250 Hz
RESAMPLING_RATE = 250

# None runs all subjects.
# [1] runs only subject 1.
# [1, 2, 15] runs only the selected subjects.
SUBJECTS = None

# TMIN, TMAX: 0-3 s for Cho2017; 0-4 s for Lee2019_MI and Yang2025_2C.

# Base output directory.
BASE = Path.home() / "WIRN26"

# Keep dataset/evaluation outputs in separate folders.
OUT_DIR = BASE / DATASET_NAME / EVAL_KIND
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = BASE / "summary_results.csv"

SEED = 23


# ===================== DL HYPERPARAMETERS ==============================

# Hyperparameters reported or aligned with the benchmark setup.
DL_BATCH_SIZE = 64  # Trials processed before each weight update.
DL_MAX_EPOCHS = 300  # Maximum number of training epochs.
DL_LR = 0.001  # Learning rate.
DL_OPTIMIZER = torch.optim.Adam  # Optimizer.
DL_PATIENCE = 75  # Early-stopping patience.
DL_AUGMENTATION = "none"  # No data augmentation.
DL_LOSS = torch.nn.CrossEntropyLoss  # Training loss.

# Additional training settings.
DL_VALID_SPLIT = 0.1  # Fraction of training data used for validation.
                     # MOABB examples commonly use 0.2.
DL_WEIGHT_DECAY = 0.0  # Weight decay regularization.
DL_LR_SCHEDULER = None  # Stored in run metadata; schedulers are set per model below.
DL_SHUFFLE = True  # Shuffle training batches at each epoch.

# Keep MOABB parallelism conservative for GPU/MPS training.
N_JOBS = 1
OVERWRITE = True


# ===================== DATASET WRAPPERS ================================

class Lee2019MI(MOABBLee2019_MI):
    # Use only session 1 from Lee2019_MI.
    # The selected configuration includes train and online-feedback test runs,
    # yielding approximately 100 trials per class in the selected session.

    def _get_single_subject_data(self, subject):
        sessions = super()._get_single_subject_data(subject)
        return {"1": sessions["1"]}


class Yang2025(MOABBYang2025):
    # Use a locally extracted Yang2025 root and avoid re-download/re-extraction.

    def __init__(self, local_root: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_root = Path(local_root)

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")

        if not self.local_root.exists():
            raise FileNotFoundError(f"Yang2025 root not found: {self.local_root}")

        if not (self.local_root / "sourcedata").exists():
            raise FileNotFoundError(
                f"This does not look like the Yang2025 extracted root: {self.local_root}"
            )

        return str(self.local_root)


# ======================== CONFIG HELPERS ================================

def safe_name(x) -> str:
    # Convert values into filesystem-safe strings.
    return (
        str(x)
        .replace(" ", "")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )

def get_dataset_time_window(dataset_name: str):
    # Return the benchmark MI time window for each dataset.

    if dataset_name == "Cho2017":
        return 0.0, 3.0

    if dataset_name == "Lee2019_MI":
        return 0.0, 4.0

    if dataset_name == "Yang2025_2C":
        return 0.0, 4.0

    raise ValueError(f"Unknown DATASET_NAME: {dataset_name}")

def make_dataset(dataset_name: str):
    # Create the dataset object and return a short metadata note.
  
    if dataset_name == "Cho2017":
        dataset = Cho2017()
        dataset_notes = "single_session"

    elif dataset_name == "Lee2019_MI":
        dataset = Lee2019MI(
            train_run=True,   # 50 trials per class.
            test_run=True,    # Additional online-feedback trials.
            resting_state=False,
        )
        dataset_notes = "session1_train_plus_online_test"

    elif dataset_name == "Yang2025_2C":
        yang_root = (
            Path.home()
            / "devEEG"
            / "MNE_DATA"
            / "MNE-yang2025-data"
            / "WBCIC_SHU Motor Imagery dataset"
        )

        dataset = Yang2025(
            local_root=yang_root,
            paradigm_type="2C",
            sessions=[1],
        )
        dataset_notes = "session1_2C_local_extracted"

    else:
        raise ValueError(f"Unknown DATASET_NAME: {dataset_name}")

    return dataset, dataset_notes

def get_session_label(dataset_name: str) -> str:
    if dataset_name == "Cho2017":
        return "0"

    if dataset_name == "Lee2019_MI":
        return "1_trainplusonline"

    if dataset_name == "Yang2025_2C":
        return "1"

    raise ValueError(f"Unknown DATASET_NAME: {dataset_name}")


# ======================== UTILITIES ====================================

def mean_std_str(series: pd.Series) -> str:
    # Return mean ± standard deviation in percentage scale.
    mean_val = series.mean() * 100
    std_val = series.std(ddof=1) * 100
    return f"{mean_val:.2f} ± {std_val:.2f}"


def save_csv_outputs(results: pd.DataFrame, exp_id: str, run_info: dict, out_dir: Path):
    # Save raw MOABB results, run metadata, and a cumulative summary file.

    # Raw MOABB results.
    raw_path = out_dir / f"{exp_id}_raw_results.csv"
    results.to_csv(raw_path, index=False, float_format="%.6f")

    # Run-level metadata.
    info_path = out_dir / f"{exp_id}_run_info.csv"
    pd.DataFrame([run_info]).to_csv(info_path, index=False)

    # Cumulative summary.
    summary_rows = []
    for pipe_name, dfp in results.groupby("pipeline"):
        summary_rows.append({
            "exp_id": exp_id,
            "evaluation": run_info["evaluation"],
            "pipeline": pipe_name,
            "score_mean_std": mean_std_str(dfp["score"]),
        })

    new_rows = pd.DataFrame(summary_rows)

    if SUMMARY_PATH.exists():
        summary_df = pd.read_csv(SUMMARY_PATH)
        summary_df = pd.concat([summary_df, new_rows], ignore_index=True)

        summary_df = summary_df.drop_duplicates(
            subset=["exp_id", "pipeline"],
            keep="last"
        )
    else:
        summary_df = new_rows

    summary_df.to_csv(SUMMARY_PATH, index=False)


def print_model_stats(name, model):
    # Optional debugging helper for initialized neural models.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{name}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(model)


def make_all_pipelines(device: str):
    # ML pipeline: TS+EL.
    ts_el = make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            max_iter=4000,
        ),
    )

    # DL pipeline: EEGNet_8_2.
    eegnet_clf = EEGClassifier(
        module=EEGNet,
        module__F1=8,
        module__D=2,
        module__kernel_length=64,
        module__drop_prob=0.25,

        criterion=DL_LOSS,
        optimizer=DL_OPTIMIZER,
        optimizer__lr=DL_LR,
        optimizer__weight_decay=DL_WEIGHT_DECAY,

        batch_size=DL_BATCH_SIZE,
        max_epochs=DL_MAX_EPOCHS,
        train_split=ValidSplit(DL_VALID_SPLIT, random_state=SEED),

        callbacks=[
            EarlyStopping(monitor="valid_loss", patience=DL_PATIENCE),
        ],

        iterator_train__shuffle=DL_SHUFFLE,
        device=device,
        verbose=1,
    )
    eegnet = make_pipeline(eegnet_clf)

    # DL pipeline: ShallowConvNet.
    shallow_clf = EEGClassifier(
        module=ShallowFBCSPNet,

        module__n_filters_time=40,
        module__filter_time_length=25,
        module__n_filters_spat=40,
        module__pool_time_length=75,
        module__pool_time_stride=15,
        module__final_conv_length="auto",
        module__pool_mode="mean",
        module__batch_norm=True,
        module__batch_norm_alpha=0.1,
        module__drop_prob=0.5,

        criterion=DL_LOSS,
        optimizer=DL_OPTIMIZER,
        optimizer__lr=DL_LR,
        optimizer__weight_decay=DL_WEIGHT_DECAY,

        batch_size=DL_BATCH_SIZE,
        max_epochs=DL_MAX_EPOCHS,
        train_split=ValidSplit(DL_VALID_SPLIT, random_state=SEED),

        callbacks=[
            EarlyStopping(monitor="valid_loss", patience=DL_PATIENCE),
            LRScheduler(
                policy=ReduceLROnPlateau,
                monitor="valid_loss",
                patience=DL_PATIENCE,
                factor=0.5,
            ),
        ],

        iterator_train__shuffle=DL_SHUFFLE,
        device=device,
        verbose=1,
    )
    shallow = make_pipeline(shallow_clf)

    # DL pipeline: DeepConvNet.
    deep_clf = EEGClassifier(
        module=Deep4Net,

        module__n_filters_time=25,
        module__n_filters_spat=25,
        module__filter_time_length=10,
        module__pool_time_length=3,
        module__pool_time_stride=3,
        module__n_filters_2=50,
        module__filter_length_2=10,
        module__n_filters_3=100,
        module__filter_length_3=10,
        module__n_filters_4=200,
        module__filter_length_4=10,
        module__drop_prob=0.5,
        module__split_first_layer=True,
        module__batch_norm=True,
        module__batch_norm_alpha=0.1,
        module__final_conv_length="auto",

        criterion=DL_LOSS,
        optimizer=DL_OPTIMIZER,
        optimizer__lr=DL_LR,
        optimizer__weight_decay=DL_WEIGHT_DECAY,

        batch_size=DL_BATCH_SIZE,
        max_epochs=DL_MAX_EPOCHS,
        train_split=ValidSplit(DL_VALID_SPLIT, random_state=SEED),

        callbacks=[
            EarlyStopping(monitor="valid_loss", patience=DL_PATIENCE),
            LRScheduler(
                policy=ReduceLROnPlateau,
                monitor="valid_loss",
                patience=DL_PATIENCE,
                factor=0.5,
            ),
        ],

        iterator_train__shuffle=DL_SHUFFLE,
        device=device,
        verbose=1,
    )
    deep = make_pipeline(deep_clf)

    return {
        "TS+EL": ts_el,
        "EEGNet_8_2": eegnet,
        "ShallowConvNet": shallow,
        "DeepConvNet": deep,
    }


# ======================== MAIN =========================================

def main():
    start = datetime.now()
    t0 = time.perf_counter()

    # ---------------------- SEED --------------------------
    setup_seed(SEED)
    torch.manual_seed(SEED)

    # ---------------------- DEVICE ------------------------
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = "cpu"

    # ---------------------- DATASET -----------------------
    dataset, dataset_notes = make_dataset(DATASET_NAME)
    session_label = get_session_label(DATASET_NAME)
    TMIN, TMAX = get_dataset_time_window(DATASET_NAME)

    if SUBJECTS is not None:
        dataset.subject_list = SUBJECTS.copy()

    # ---------------------- PARADIGM ----------------------
    paradigm = MotorImagery(
        n_classes=2,
        events=["left_hand", "right_hand"],
        fmin=8.0,
        fmax=32.0,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        resample=RESAMPLING_RATE,
    )

    # ---------------------- PIPELINE ----------------------
    all_pipelines = make_all_pipelines(device)

    if PIPELINE_NAME not in all_pipelines:
        raise ValueError(
            f"Unknown PIPELINE_NAME: {PIPELINE_NAME}. "
            f"Available: {list(all_pipelines.keys())}"
        )

    pipelines = {
        PIPELINE_NAME: all_pipelines[PIPELINE_NAME]
    }

    # ---------------------- EVALUATION --------------------
    suffix = (
        f"{safe_name(dataset.code)}_"
        f"{EVAL_KIND}_"
        f"sess{safe_name(session_label)}_"
        f"t{TMIN}-{TMAX}_"
        f"sr{safe_name(RESAMPLING_RATE)}_"
        f"{safe_name(PIPELINE_NAME)}_"
        f"seed{SEED}"
    )

    if EVAL_KIND == "within":
        evaluation = WithinSessionEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            random_state=SEED,
            n_jobs=N_JOBS,
            overwrite=OVERWRITE,
            suffix=suffix,
        )
    elif EVAL_KIND == "cross":
        evaluation = CrossSubjectEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            random_state=SEED,
            n_jobs=N_JOBS,
            overwrite=OVERWRITE,
            suffix=suffix,
        )
    else:
        raise ValueError("EVAL_KIND must be 'within' or 'cross'")

    # ---------------------- RUN ID ------------------------
    # Example: 20260325_100509_Cho2017_within_TS+EL
    exp_id = (
        f"{start:%Y%m%d_%H%M%S}_"
        f"{safe_name(DATASET_NAME)}_"
        f"{EVAL_KIND}_"
        f"{safe_name(PIPELINE_NAME)}"
    )

    # ---------------------- PRINT INFO --------------------
    print(f"Dataset name: {DATASET_NAME}")
    print(f"Dataset code: {dataset.code}")
    print(f"Dataset notes: {dataset_notes}")
    print(f"Evaluation: {EVAL_KIND}")
    print(f"Sessions: {session_label}")
    print(f"Subjects: {dataset.subject_list}")
    print(f"Pipelines: {list(pipelines.keys())}")
    print("Band: 8.0-32.0 Hz")
    print(f"Epoch: {TMIN}-{TMAX} s")
    print(f"Sample rate: {RESAMPLING_RATE} Hz")
    print(f"Start: {start:%Y-%m-%d %H:%M:%S}")
    print("MPS available:", torch.backends.mps.is_available())
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)

    # ---------------------- RUN ---------------------------
    # MOABB returns a DataFrame with score, timing, subject, dataset, and pipeline columns.
    results = evaluation.process(pipelines) 

    # ---------------------- RUN INFO ----------------------
    # Store metadata required to reproduce the run.
    run_info = {
        "exp_id": exp_id,
        "dataset_name": DATASET_NAME,
        "dataset": dataset.code,
        "dataset_notes": dataset_notes,
        "evaluation": EVAL_KIND,
        "sessions": session_label,
        "seed": SEED,
        "subjects": ",".join(map(str, dataset.subject_list)),
        "metric": "roc_auc",
        "events": "left_hand,right_hand",
        "fmin": 8.0,
        "fmax": 32.0,
        "tmin": TMIN,
        "tmax": TMAX,
        "SR": RESAMPLING_RATE,
        "pipelines": ",".join(pipelines.keys()),
        "pipeline_selected": PIPELINE_NAME,
        "dl_optimizer": "Adam",
        "dl_loss": "CrossEntropyLoss",
        "dl_lr": DL_LR,
        "dl_batch_size": DL_BATCH_SIZE,
        "dl_max_epochs": DL_MAX_EPOCHS,
        "dl_patience": DL_PATIENCE,
        "dl_weight_decay": DL_WEIGHT_DECAY,
        "dl_augmentation": DL_AUGMENTATION,
        "dl_valid_split": DL_VALID_SPLIT,
        "dl_shuffle": DL_SHUFFLE,
        "dl_lr_scheduler": str(DL_LR_SCHEDULER),
        "device": device,
        "moabb_version": moabb.__version__,
        "mne_version": mne.__version__,
        "torch_version": torch.__version__,
    }

    save_csv_outputs(results, exp_id, run_info, OUT_DIR)

    # ---------------------- END PRINT ---------------------
    elapsed_s = time.perf_counter() - t0

    print(f"Saved CSVs for: {exp_id}")
    print(f"Elapsed time: {elapsed_s:.2f} s ({elapsed_s / 60:.2f} min)")
    print("Pipelines:", list(pipelines.keys()))
    print("Subjects:", dataset.subject_list)


if __name__ == "__main__":
    main()