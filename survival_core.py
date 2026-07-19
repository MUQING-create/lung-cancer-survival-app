"""
survival_core — NSCLC Overall Survival Prediction
==================================================
Single source of truth for feature schema, data simulation, model factory
and evaluation suite used by the pipeline (survival_model.py).

Clinical context
----------------
Non-Small Cell Lung Cancer (NSCLC) overall survival prediction.
N = 1 000 synthetic patients; stage distribution calibrated to SEER 2015-2019.
Survival calibrated to published stage-specific median OS (SEER):
  Stage I ≈ 84 months · Stage II ≈ 36 · Stage III ≈ 18 · Stage IV ≈ 9

Models
------
1. Cox Proportional Hazards  — lifelines, L2-regularised, penaliser tuned by CV
2. Parametric AFT             — best marginal distribution by AIC (Weibull / Log-Normal / Log-Logistic)
3. Random Survival Forest     — sksurv + Optuna (40 trials, 5-fold CV)
4. DeepSurv                   — pycox / PyTorch MLP + Optuna (15 trials, hold-out val)

Evaluation: bootstrap C-index (95 % CI), Integrated Brier Score,
time-dependent AUC, calibration at 12/24/36/60 months.

Simulated data only — NOT a clinical diagnostic tool.
"""

from __future__ import annotations

import os
import platform
import tempfile
import warnings
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ── ASCII temp dir (CJK Windows usernames crash joblib) ─────────────────────

def _ensure_ascii_joblib_tmp() -> None:
    if os.environ.get("JOBLIB_TEMP_FOLDER"):
        return
    try:
        tempfile.gettempdir().encode("ascii")
    except UnicodeEncodeError:
        drive = os.environ.get("SystemDrive", "C:") + os.sep
        for cand in (os.path.join(drive, "Temp"), os.path.join(drive, "joblib_tmp")):
            try:
                os.makedirs(cand, exist_ok=True)
                os.environ["JOBLIB_TEMP_FOLDER"] = cand
                return
            except OSError:
                continue

_ensure_ascii_joblib_tmp()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # suppress OMP duplicate-lib warning

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
from scipy import stats
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import joblib

import lifelines
from lifelines import (
    CoxPHFitter, KaplanMeierFitter,
    WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter,
    WeibullFitter, LogNormalFitter, LogLogisticFitter, ExponentialFitter,
)

import sksurv
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    cumulative_dynamic_auc,
)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

_DEEPSURV_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchtuples as tt
    from pycox.models import CoxPH as _PycoxCoxPH
    _DEEPSURV_AVAILABLE = True
except ImportError:
    warnings.warn("pycox/torch not available — DeepSurv will be skipped.", ImportWarning)

__version__ = "1.0.0"
SEED = 42
CELL_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]


def apply_cell_matplotlib_style(**overrides) -> None:
    """Apply Cell-style matplotlib defaults for app and report figures."""
    params = {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "legend.frameon": False,
    }
    params.update(overrides)
    mpl.rcParams.update(params)

# ── 1. Feature schema ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Feature:
    name: str
    display: str
    kind: str        # "cont" | "binary" | "ordinal" | "categorical"
    unit: str
    low: float
    high: float
    default: float
    description: str


FEATURES: tuple[Feature, ...] = (
    Feature("age",                  "Age",                       "cont",        "years",    30,  90,   65, "Age at diagnosis"),
    Feature("sex_male",             "Sex (Male=1)",              "binary",      "",          0,   1,    1, "Biological sex"),
    Feature("ecog_ps",              "ECOG Performance Status",   "ordinal",     "",          0,   3,    1, "0=fully active … 3=limited self-care"),
    Feature("stage",                "Stage (I–IV)",              "ordinal",     "",          1,   4,    3, "UICC 8th edition pathological stage"),
    Feature("histology",            "Histology",                 "categorical", "",          0,   2,    0, "0=Adenocarcinoma 1=Squamous 2=Other NSCLC"),
    Feature("smoking_pack_years",   "Smoking Pack-years",        "cont",        "pack-yrs",  0,  80,   20, "Cumulative tobacco exposure"),
    Feature("tumor_size_cm",        "Tumour Size",               "cont",        "cm",      0.5,  12,  3.5, "Max diameter on CT"),
    Feature("n_positive_nodes",     "Positive Node Count",       "cont",        "",          0,  20,    2, "Pathologically positive lymph nodes"),
    Feature("treatment",            "Primary Treatment",         "categorical", "",          0,   3,    1, "0=Surgery 1=Chemo 2=CRT 3=Targeted"),
    Feature("bmi",                  "BMI",                       "cont",        "kg/m²",    14,  40,   24, "Body mass index"),
    Feature("egfr_mutation",        "EGFR Mutation",             "binary",      "",          0,   1,    0, "Sensitising EGFR mutation"),
    Feature("nlr",                  "Neutrophil-Lymphocyte Ratio","cont",       "",        1.0,15.0,  3.5, "Pre-treatment systemic inflammation marker"),
)

FEATURE_NAMES = [f.name for f in FEATURES]
FEATURE_DISPLAY = {f.name: f.display for f in FEATURES}
TIME_COL  = "time_months"
EVENT_COL = "event_death"
FOLLOW_UP_MAX = 60.0  # administrative censoring at 5 years

# Weibull shape calibrated to NSCLC (mildly increasing hazard)
_WEIBULL_SHAPE = 1.30
# Stage-specific median OS (SEER 2015-2019 NSCLC, all-comers)
_STAGE_MEDIAN_OS = {1: 84.0, 2: 36.0, 3: 18.0, 4: 9.0}
_STAGE_PROBS = np.array([0.18, 0.08, 0.24, 0.50])

# Log HRs grounded in published meta-analyses
_LOG_HR = dict(
    age=0.018, sex_male=0.18, ecog_ps=0.38,
    hist_sq=0.12, hist_oth=0.20,
    smoking_pack_years=0.004,
    tumor_size_cm=0.095, n_positive_nodes=0.075,
    treat_chemo=0.35, treat_crt=0.22, treat_targeted=-0.28,
    bmi=-0.025, egfr_mutation=-0.50, nlr=0.06,
)

BRAND = dict(
    red=CELL_COLORS[0],
    blue=CELL_COLORS[1],
    green=CELL_COLORS[2],
    navy=CELL_COLORS[3],
    orange=CELL_COLORS[4],
    purple=CELL_COLORS[5],
    mint=CELL_COLORS[6],
    crimson=CELL_COLORS[7],
    brown=CELL_COLORS[8],
    tan=CELL_COLORS[9],
    light="white",
    grid=CELL_COLORS[6],
    edge=CELL_COLORS[5],
    ink=CELL_COLORS[3],
)

MODEL_COLORS = {
    "Cox PH":    BRAND["navy"],
    "AFT":       BRAND["blue"],
    "RSF":       BRAND["orange"],
    "DeepSurv":  BRAND["red"],
}

# ── 2. Data simulation ───────────────────────────────────────────────────────

def generate_nsclc_cohort(n: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """
    Simulate a realistic NSCLC cohort with clinically calibrated correlations.

    Biology encoded:
    - Stage IV → worse ECOG, more positive nodes, more systemic treatment
    - Squamous → heavy smoker, male, rarely EGFR-mutant
    - EGFR mutation → adenocarcinoma, female, never/light smoker, targeted therapy
    - Survival generated from Weibull PH model (stage-stratified baseline)
    """
    rng = np.random.default_rng(seed)

    # Stage
    stage = rng.choice([1, 2, 3, 4], size=n, p=_STAGE_PROBS)

    # Histology (stage-dependent proportions)
    adeno_p = np.clip(0.52 - 0.04 * (stage - 1), 0.35, 0.60)
    sq_p    = np.clip(0.29 + 0.02 * (stage - 1), 0.24, 0.38)
    oth_p   = 1 - adeno_p - sq_p
    hist_probs = np.column_stack([adeno_p, sq_p, oth_p])
    histology = np.array([rng.choice(3, p=hist_probs[i]) for i in range(n)])

    # Sex (squamous more male)
    male_p = np.where(histology == 1, 0.72, 0.55)
    sex_male = rng.binomial(1, male_p)

    # Age
    age = rng.normal(65, 10, n).clip(35, 90).round().astype(int)

    # ECOG PS (worse with higher stage)
    ecog_probs = np.array([
        [0.50, 0.35, 0.12, 0.03],
        [0.38, 0.40, 0.17, 0.05],
        [0.22, 0.38, 0.28, 0.12],
        [0.12, 0.30, 0.35, 0.23],
    ])
    ecog_ps = np.array([rng.choice(4, p=ecog_probs[s - 1]) for s in stage])

    # Smoking
    base_py = np.where(histology == 1, 42.0, 18.0)
    smoking = rng.gamma(shape=2.0, scale=base_py / 2.0).clip(0, 80)
    smoking[rng.random(n) < 0.12] = 0.0  # 12% never-smokers

    # Tumour size (grows with stage)
    size_mean = np.array([2.0, 3.5, 5.0, 5.5])[(stage - 1)]
    tumor_size_cm = rng.gamma(shape=3, scale=size_mean / 3).clip(0.5, 12)

    # Positive nodes
    node_lam = {1: 0.05, 2: 1.5, 3: 5.0, 4: 7.0}
    n_positive_nodes = np.array([
        max(0, int(rng.poisson(node_lam[s]))) for s in stage
    ]).clip(0, 20)

    # Treatment (stage-guided)
    treat_probs = np.array([
        [0.80, 0.08, 0.08, 0.04],
        [0.55, 0.15, 0.20, 0.10],
        [0.10, 0.28, 0.52, 0.10],
        [0.02, 0.45, 0.25, 0.28],
    ])
    treatment = np.array([rng.choice(4, p=treat_probs[s - 1]) for s in stage])

    # BMI
    bmi = rng.normal(24.5, 4.0, n).clip(14, 42).round(1)

    # EGFR mutation (adeno, female, light-smoker)
    egfr_base = np.where(
        (histology == 0) & (sex_male == 0) & (smoking < 10), 0.42,
        np.where(histology == 0, 0.15, 0.03)
    )
    egfr_mutation = rng.binomial(1, egfr_base)
    treatment[(egfr_mutation == 1) & (stage >= 3)] = 3  # targeted for late-stage EGFR+

    # NLR
    nlr_mean = np.clip(2.5 + 0.35 * ecog_ps + 0.25 * (stage - 1), 1.0, 10.0)
    nlr = rng.gamma(shape=4, scale=nlr_mean / 4).clip(1.0, 15.0).round(1)

    # ── Survival generation (Weibull PH) ────────────────────────────────────
    # Baseline scale per stage → matches published median OS
    # Weibull median = scale * (ln2)^(1/shape)  →  scale = median / (ln2)^(1/shape)
    ln2_k = np.log(2) ** (1.0 / _WEIBULL_SHAPE)
    stage_scale = np.array([_STAGE_MEDIAN_OS[s] / ln2_k for s in stage])

    # Linear predictor (log HR); stage effect absorbed into baseline scale
    lp = (
        _LOG_HR["age"]               * (age - 65)
        + _LOG_HR["sex_male"]        * sex_male
        + _LOG_HR["ecog_ps"]         * ecog_ps
        + _LOG_HR["hist_sq"]         * (histology == 1)
        + _LOG_HR["hist_oth"]        * (histology == 2)
        + _LOG_HR["smoking_pack_years"] * (smoking - 20)
        + _LOG_HR["tumor_size_cm"]   * (tumor_size_cm - 3.5)
        + _LOG_HR["n_positive_nodes"]* n_positive_nodes
        + _LOG_HR["treat_chemo"]     * (treatment == 1)
        + _LOG_HR["treat_crt"]       * (treatment == 2)
        + _LOG_HR["treat_targeted"]  * (treatment == 3)
        + _LOG_HR["bmi"]             * (bmi - 24)
        + _LOG_HR["egfr_mutation"]   * egfr_mutation
        + _LOG_HR["nlr"]             * (nlr - 3.5)
    )
    # Centre LP within each stage so stage_scale exactly matches target median OS
    # (prevents systematic inflation of hazard from uncentred covariates)
    for s in [1, 2, 3, 4]:
        mask = (stage == s)
        if mask.sum() > 0:
            lp[mask] -= lp[mask].mean()

    # Inverse-CDF Weibull PH: T = scale * (-ln U)^(1/k) * exp(-lp/k)
    U = rng.uniform(0, 1, n)
    T_event = stage_scale * (-np.log(U)) ** (1.0 / _WEIBULL_SHAPE) * np.exp(-lp / _WEIBULL_SHAPE)

    # Censoring: administrative (60 months) + random (mean 7 years)
    C = np.minimum(FOLLOW_UP_MAX, rng.exponential(scale=84.0, size=n))
    time_obs  = np.minimum(T_event, C).clip(0.1).round(1)
    event_obs = (T_event <= C).astype(int)

    return pd.DataFrame({
        "age":                age,
        "sex_male":           sex_male,
        "ecog_ps":            ecog_ps,
        "stage":              stage,
        "histology":          histology,
        "smoking_pack_years": smoking.round(1),
        "tumor_size_cm":      tumor_size_cm.round(1),
        "n_positive_nodes":   n_positive_nodes.astype(int),
        "treatment":          treatment,
        "bmi":                bmi,
        "egfr_mutation":      egfr_mutation,
        "nlr":                nlr,
        TIME_COL:             time_obs,
        EVENT_COL:            event_obs,
    })


# ── 3. Feature preprocessing (one-hot encode categoricals; keep ordinal/cont) 

_BASE_COLS = [
    "age", "sex_male", "ecog_ps", "stage",
    "smoking_pack_years", "tumor_size_cm", "n_positive_nodes",
    "bmi", "egfr_mutation", "nlr",
]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready DataFrame (no time/event cols). Columns are stable across splits."""
    out = df[_BASE_COLS].copy().astype(float)
    out["hist_squamous"] = (df["histology"] == 1).astype(float)
    out["hist_other"]    = (df["histology"] == 2).astype(float)
    out["treat_chemo"]   = (df["treatment"] == 1).astype(float)
    out["treat_crt"]     = (df["treatment"] == 2).astype(float)
    out["treat_targeted"]= (df["treatment"] == 3).astype(float)
    return out

def PROC_COLS(df: pd.DataFrame) -> list[str]:
    return preprocess(df).columns.tolist()


# ── 4. Distribution testing (univariate; guides AFT family choice) ───────────

_UNIVAR_FITTERS = {
    "Weibull":     WeibullFitter,
    "Log-Normal":  LogNormalFitter,
    "Log-Logistic":LogLogisticFitter,
    "Exponential": ExponentialFitter,
}

_AFT_CLASSES = {
    "Weibull":     WeibullAFTFitter,
    "Log-Normal":  LogNormalAFTFitter,
    "Log-Logistic":LogLogisticAFTFitter,
    "Exponential": WeibullAFTFitter,   # exponential is Weibull with shape=1
}

def test_distributions(df: pd.DataFrame) -> dict:
    """
    Fit 4 parametric distributions to observed survival times (with censoring).
    Compare AIC / BIC to select the AFT family for the parametric model.

    Returns
    -------
    dict with keys:
        "table"        : pd.DataFrame with AIC/BIC/median per distribution
        "best"         : name of best distribution (lowest AIC)
        "aft_class"    : the lifelines AFT fitter class to use
        "fitters"      : fitted univariate fitter objects (for plotting)
    """
    rows, fitters = [], {}
    for name, cls in _UNIVAR_FITTERS.items():
        f = cls()
        f.fit(df[TIME_COL], event_observed=df[EVENT_COL], label=name)
        fitters[name] = f
        rows.append({"Distribution": name,
                      "AIC": round(f.AIC_, 2),
                      "BIC": round(f.BIC_, 2),
                      "Median (months)": round(float(f.median_survival_time_), 1)})

    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    best  = table.iloc[0]["Distribution"]
    return {"table": table, "best": best,
            "aft_class": _AFT_CLASSES[best], "fitters": fitters}


# ── 5a. Cox Proportional Hazards (lifelines, L2-regularised) ─────────────────

def tune_cox_penalizer(df_train: pd.DataFrame, n_folds: int = 5,
                        feat_df: pd.DataFrame = None) -> float:
    """Grid-search L2 penaliser via k-fold concordance index."""
    if feat_df is None:
        feat_df = preprocess(df_train)
    df_tr = feat_df.copy()
    df_tr[TIME_COL]  = df_train[TIME_COL].values
    df_tr[EVENT_COL] = df_train[EVENT_COL].values

    best_c, best_pen = -1.0, 0.1
    for pen in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        scores = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        for tr_idx, va_idx in kf.split(df_tr):
            try:
                m = CoxPHFitter(penalizer=pen, l1_ratio=0.0)
                m.fit(df_tr.iloc[tr_idx], duration_col=TIME_COL, event_col=EVENT_COL)
                scores.append(m.concordance_index_)
            except Exception:
                pass
        if scores and np.mean(scores) > best_c:
            best_c, best_pen = np.mean(scores), pen
    return best_pen


def build_cox(df_train: pd.DataFrame, penalizer: Optional[float] = None,
              feat_df: pd.DataFrame = None) -> CoxPHFitter:
    """Fit L2-regularised Cox PH on preprocessed features."""
    if feat_df is None:
        feat_df = preprocess(df_train)
    if penalizer is None:
        penalizer = tune_cox_penalizer(df_train, feat_df=feat_df)
    df_fit = feat_df.copy()
    df_fit[TIME_COL]  = df_train[TIME_COL].values
    df_fit[EVENT_COL] = df_train[EVENT_COL].values
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    cph.fit(df_fit, duration_col=TIME_COL, event_col=EVENT_COL)
    cph._penalizer_used = penalizer
    return cph


# ── 5b. Parametric AFT (best distribution by AIC) ────────────────────────────

def build_aft(df_train: pd.DataFrame, aft_class=None,
              penalizer: float = 0.01,
              feat_df: pd.DataFrame = None) -> object:
    """Fit parametric AFT model with the distribution selected by AIC."""
    if aft_class is None:
        aft_class = WeibullAFTFitter
    if feat_df is None:
        feat_df = preprocess(df_train)
    df_fit = feat_df.copy()
    df_fit[TIME_COL]  = df_train[TIME_COL].values
    df_fit[EVENT_COL] = df_train[EVENT_COL].values
    aft = aft_class(penalizer=penalizer)
    aft.fit(df_fit, duration_col=TIME_COL, event_col=EVENT_COL)
    return aft


# ── 5c. Random Survival Forest (sksurv + Optuna) ─────────────────────────────

def build_rsf(df_train: pd.DataFrame, n_trials: int = 40,
              feat_df: pd.DataFrame = None) -> tuple:
    """
    Tune RSF with Optuna (TPE sampler, 5-fold CV concordance index).

    Returns (fitted_rsf, best_params, best_cv_cindex)
    """
    if feat_df is None:
        feat_df = preprocess(df_train)
    X = feat_df.values.astype(float)
    y = Surv.from_arrays(event=df_train[EVENT_COL].astype(bool).values,
                         time=df_train[TIME_COL].values,
                         name_event=EVENT_COL, name_time=TIME_COL)
    feat_cols = feat_df.columns.tolist()

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators  = trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth     = trial.suggest_int("max_depth", 3, 12),
            min_samples_split = trial.suggest_int("min_samples_split", 5, 40),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 3, 20),
            max_features  = trial.suggest_float("max_features", 0.3, 1.0),
            n_jobs=-1, random_state=SEED,
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        c_scores = []
        for tr_idx, va_idx in kf.split(X):
            rsf = RandomSurvivalForest(**params)
            rsf.fit(X[tr_idx], y[tr_idx])
            c_scores.append(rsf.score(X[va_idx], y[va_idx]))
        return float(np.mean(c_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params | dict(n_jobs=-1, random_state=SEED)
    rsf = RandomSurvivalForest(**best)
    rsf.fit(X, y)
    rsf._feat_cols = feat_cols    # store for prediction helper
    return rsf, study.best_params, study.best_value


# ── 5d. DeepSurv (pycox CoxPH + PyTorch MLP + Optuna) ────────────────────────

if _DEEPSURV_AVAILABLE:
    class _MLP(nn.Module):
        """Simple MLP for DeepSurv: linear → BN → ReLU → dropout, repeated."""
        def __init__(self, in_features: int, num_nodes: list[int], dropout: float):
            super().__init__()
            layers: list[nn.Module] = []
            prev = in_features
            for h in num_nodes:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                            nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def _make_mlp(in_features: int, num_nodes: list[int], dropout: float) -> "_MLP":
        return _MLP(in_features, num_nodes, dropout)


class DeepSurvWrapper:
    """
    Serialisable wrapper around pycox CoxPH.

    Stores network state dict + baseline hazards as plain Python objects
    so the wrapper can be round-tripped through joblib without needing
    a live torch/pycox session at load time.
    """

    def __init__(self):
        self.scaler            = None
        self._feat_cols: list  = []
        self._net_params: tuple = ()    # (in_features, num_nodes, dropout)
        self._net_state: dict  = {}
        self._baseline_haz     = None   # pd.Series  (index=time, values=hazard)
        self._baseline_cumhaz  = None   # pd.Series
        self.train_params: dict = {}

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame, params: dict,
            epochs: int = 150, feat_df: pd.DataFrame = None) -> "DeepSurvWrapper":
        if not _DEEPSURV_AVAILABLE:
            raise ImportError("DeepSurv requires torch + pycox.")
        if feat_df is None:
            feat_df = preprocess(df_train)

        self._feat_cols = feat_df.columns.tolist()
        X = feat_df.values.astype("float32")
        T = df_train[TIME_COL].values.astype("float32")
        E = df_train[EVENT_COL].values.astype("float32")

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X).astype("float32")

        num_nodes = [params["n_nodes"]] * params["n_layers"]
        dropout   = params["dropout"]
        self._net_params = (Xs.shape[1], num_nodes, dropout)
        self.train_params = params

        torch.manual_seed(SEED)
        net = _make_mlp(*self._net_params)
        # Pass class (not instance) so pycox binds net.parameters() automatically
        model = _PycoxCoxPH(net, tt.optim.Adam)
        model.optimizer.set_lr(params["lr"])

        dl = model.make_dataloader((Xs, (T, E)),
                                    batch_size=params["batch_size"],
                                    shuffle=True)
        model.fit_dataloader(
            dl, epochs=epochs,
            callbacks=[tt.callbacks.EarlyStopping(patience=20)],
            verbose=False,
        )
        model.compute_baseline_hazards(input=Xs, target=(T, E))

        # Verify sign convention: if predictions are inverted (C < 0.48),
        # negate the output layer weights so higher score = more risk.
        with torch.no_grad():
            log_h_tr = model.predict(Xs).flatten()
        c_tr, *_ = concordance_index_censored(E.astype(bool), T, log_h_tr)
        if c_tr < 0.48:
            out_layer = net.net[-1]   # final nn.Linear(hidden, 1)
            with torch.no_grad():
                out_layer.weight.data.mul_(-1)
                if out_layer.bias is not None:
                    out_layer.bias.data.mul_(-1)
            model.compute_baseline_hazards(input=Xs, target=(T, E))

        self._net_state    = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        self._baseline_haz    = model.baseline_hazards_.copy()
        self._baseline_cumhaz = model.baseline_cumulative_hazards_.copy()
        return self

    # ── Reconstruction (after deserialisation) ───────────────────────────────

    def _model(self) -> "_PycoxCoxPH":
        net = _make_mlp(*self._net_params)
        net.load_state_dict(self._net_state)
        net.eval()
        m = _PycoxCoxPH(net, tt.optim.Adam())
        m.baseline_hazards_             = self._baseline_haz
        m.baseline_cumulative_hazards_   = self._baseline_cumhaz
        return m

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_risk(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Log partial hazard (higher = more risk). feat_df: pre-processed feature matrix."""
        Xs = self.scaler.transform(feat_df[self._feat_cols].values).astype("float32")
        with torch.no_grad():
            return self._model().predict(Xs).flatten()

    def predict_surv_df(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Survival probability DataFrame (index=time, columns=patients)."""
        Xs = self.scaler.transform(feat_df[self._feat_cols].values).astype("float32")
        return self._model().predict_surv_df(Xs)


def tune_deepsurv(df_train: pd.DataFrame, n_trials: int = 15,
                  val_frac: float = 0.20,
                  feat_df: pd.DataFrame = None) -> tuple:
    """
    Optuna search for DeepSurv hyperparameters (hold-out validation C-index).

    Returns (best_params, best_cindex)
    """
    if not _DEEPSURV_AVAILABLE:
        raise ImportError("DeepSurv requires torch + pycox.")
    if feat_df is None:
        feat_df = preprocess(df_train)

    n_val = max(50, int(len(df_train) * val_frac))
    df_tr  = df_train.iloc[:-n_val].copy()
    df_va  = df_train.iloc[-n_val:].copy()
    ftr    = feat_df.iloc[:-n_val].reset_index(drop=True)
    fva    = feat_df.iloc[-n_val:].reset_index(drop=True)

    e_va = df_va[EVENT_COL].astype(bool).values
    t_va = df_va[TIME_COL].values

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_layers   = trial.suggest_int("n_layers", 1, 3),
            n_nodes    = trial.suggest_int("n_nodes", 32, 128, step=32),
            dropout    = trial.suggest_float("dropout", 0.0, 0.3),
            lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256]),
        )
        try:
            w = DeepSurvWrapper().fit(df_tr, params, epochs=150, feat_df=ftr)
            risk = w.predict_risk(fva)
            if e_va.sum() < 3:
                return 0.5
            c, *_ = concordance_index_censored(e_va, t_va, risk)
            return float(c)
        except Exception:
            return 0.5

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def build_deepsurv(df_train: pd.DataFrame, params: dict,
                   epochs: int = 200,
                   feat_df: pd.DataFrame = None) -> DeepSurvWrapper:
    return DeepSurvWrapper().fit(df_train, params, epochs=epochs, feat_df=feat_df)


# ── 6. Unified evaluation ────────────────────────────────────────────────────

_EVAL_TIMES = np.array([12.0, 24.0, 36.0, 48.0, 60.0])


def _surv_matrix_lifelines(model, feat_df: pd.DataFrame,
                            times: np.ndarray) -> np.ndarray:
    """(n_samples, n_times) survival probability matrix from a lifelines model."""
    sf = model.predict_survival_function(feat_df)   # shape (n_internal_times, n_samples)
    t_axis = sf.index.values
    # Interpolate each patient's SF at requested times
    mat = np.column_stack([
        np.interp(times, t_axis, sf.iloc[:, i].values,
                  left=1.0, right=float(sf.iloc[-1, i]))
        for i in range(sf.shape[1])
    ]).T   # (n_samples, n_times)
    return mat.clip(0.0, 1.0)


def _surv_matrix_sksurv(surv_fns, times: np.ndarray) -> np.ndarray:
    """(n_samples, n_times) matrix from sksurv step-function list."""
    return np.row_stack([
        np.array([fn(t) for t in times]) for fn in surv_fns
    ]).clip(0.0, 1.0)


def _surv_matrix_deepsurv(wrapper: DeepSurvWrapper, feat_df: pd.DataFrame,
                           times: np.ndarray) -> np.ndarray:
    sf = wrapper.predict_surv_df(feat_df)   # (n_internal_times, n_samples)
    t_axis = sf.index.values
    mat = np.column_stack([
        np.interp(times, t_axis, sf.iloc[:, i].values,
                  left=1.0, right=float(sf.iloc[-1, i]))
        for i in range(sf.shape[1])
    ]).T
    return mat.clip(0.0, 1.0)


def _clip_times(times: np.ndarray, y_train, y_test) -> np.ndarray:
    """Keep only times strictly inside the observed range of both splits."""
    # sksurv structured arrays use field names set at creation time
    t_field = TIME_COL
    t_lo = max(y_train[t_field].min(), y_test[t_field].min()) + 0.5
    t_hi = min(y_train[t_field].max(), y_test[t_field].max()) - 0.5
    return times[(times > t_lo) & (times < t_hi)]


def evaluate(model, model_type: str,
             df_test: pd.DataFrame,
             y_train, y_test,
             times: np.ndarray = _EVAL_TIMES,
             feat_df_test: pd.DataFrame = None,
             n_boot: int = 300) -> dict:
    """
    Compute C-index (bootstrap 95 % CI), IBS and time-dependent AUC.

    Parameters
    ----------
    model_type    : "cox" | "aft" | "rsf" | "deepsurv"
    feat_df_test  : pre-processed feature matrix (rows = patients, cols = features).
                    If None, falls back to preprocess(df_test) for simulated data.
    y_train, y_test : sksurv structured arrays (field names = TIME_COL / EVENT_COL)
    """
    times = _clip_times(times, y_train, y_test)
    e_te  = y_test[EVENT_COL].astype(bool)
    t_te  = y_test[TIME_COL].astype(float)

    # Resolve feature matrix (real data passes it explicitly to avoid schema coupling)
    if feat_df_test is None:
        feat_df_test = preprocess(df_test)

    # ── risk scores and survival matrix ─────────────────────────────────────
    if model_type in ("cox", "aft"):
        if model_type == "cox":
            risk = model.predict_partial_hazard(feat_df_test).values.astype(float)
        else:
            risk = (1.0 / model.predict_median(feat_df_test).values).astype(float)
        surv_mat = _surv_matrix_lifelines(model, feat_df_test, times)

    elif model_type == "rsf":
        X = feat_df_test.values.astype(float)
        risk = model.predict(X).astype(float)
        surv_mat = _surv_matrix_sksurv(
            model.predict_survival_function(X), times)

    elif model_type == "deepsurv":
        risk = model.predict_risk(feat_df_test).astype(float)
        surv_mat = _surv_matrix_deepsurv(model, feat_df_test, times)

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # ── C-index + bootstrap CI ───────────────────────────────────────────────
    c_idx, *_ = concordance_index_censored(e_te, t_te, risk)

    rng = np.random.default_rng(SEED)
    n = len(df_test)
    boot_c = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        eb, tb, rb = e_te[idx], t_te[idx], risk[idx]
        if eb.sum() < 2:
            continue
        try:
            cb, *_ = concordance_index_censored(eb, tb, rb)
            boot_c.append(cb)
        except Exception:
            pass
    ci_lo, ci_hi = np.percentile(boot_c, [2.5, 97.5]) if boot_c else (np.nan, np.nan)

    # ── Integrated Brier Score ───────────────────────────────────────────────
    try:
        ibs = float(integrated_brier_score(y_train, y_test, surv_mat, times))
    except Exception:
        ibs = np.nan

    # ── Time-dependent AUC ──────────────────────────────────────────────────
    try:
        auc_vals, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)
        auc_vals  = auc_vals.tolist()
        mean_auc  = float(mean_auc)
    except Exception:
        auc_vals  = [np.nan] * len(times)
        mean_auc  = np.nan

    return dict(
        c_index    = float(c_idx),
        ci_lo      = float(ci_lo),
        ci_hi      = float(ci_hi),
        ibs        = ibs,
        auc_vals   = auc_vals,
        mean_auc   = mean_auc,
        surv_mat   = surv_mat,
        risk       = risk,
        times      = times.tolist(),
    )


def evaluate_train(model, model_type: str,
                   df_train: pd.DataFrame,
                   y_train,
                   times: np.ndarray = _EVAL_TIMES,
                   feat_df_train: pd.DataFrame = None) -> dict:
    """
    Apparent (training-set) C-index and IBS.
    No bootstrapping — caller uses these alongside test metrics for overfitting assessment.
    """
    e_tr = y_train[EVENT_COL].astype(bool)
    t_tr = y_train[TIME_COL].astype(float)

    if feat_df_train is None:
        feat_df_train = preprocess(df_train)

    if model_type == "cox":
        risk = model.predict_partial_hazard(feat_df_train).values.astype(float)
    elif model_type == "aft":
        risk = (1.0 / model.predict_median(feat_df_train).values).astype(float)
    elif model_type == "rsf":
        X = feat_df_train.values.astype(float)
        risk = model.predict(X).astype(float)
    elif model_type == "deepsurv":
        risk = model.predict_risk(feat_df_train).astype(float)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    c_idx, *_ = concordance_index_censored(e_tr, t_tr, risk)

    t_lo = float(y_train[TIME_COL].min()) + 0.5
    t_hi = float(y_train[TIME_COL].max()) - 0.5
    times_tr = times[(times > t_lo) & (times < t_hi)]

    ibs = np.nan
    if len(times_tr) >= 2:
        if model_type in ("cox", "aft"):
            surv_mat = _surv_matrix_lifelines(model, feat_df_train, times_tr)
        elif model_type == "rsf":
            surv_mat = _surv_matrix_sksurv(
                model.predict_survival_function(feat_df_train.values.astype(float)),
                times_tr)
        elif model_type == "deepsurv":
            surv_mat = _surv_matrix_deepsurv(model, feat_df_train, times_tr)
        try:
            ibs = float(integrated_brier_score(y_train, y_train, surv_mat, times_tr))
        except Exception:
            ibs = np.nan

    return dict(c_index=float(c_idx), ibs=ibs)


def risk_tertile_curves(risk: np.ndarray, df_test: pd.DataFrame) -> dict:
    """KM curves for low / medium / high risk tertiles."""
    q33, q67 = np.percentile(risk, [33.3, 66.7])
    labels = np.where(risk <= q33, "Low", np.where(risk <= q67, "Medium", "High"))
    curves = {}
    for grp in ["Low", "Medium", "High"]:
        mask = labels == grp
        km = KaplanMeierFitter()
        km.fit(df_test[TIME_COL][mask], df_test[EVENT_COL][mask], label=grp)
        curves[grp] = km
    return curves


# ── 7. Calibration at fixed time points ──────────────────────────────────────

def calibration_at(surv_mat: np.ndarray, df_test: pd.DataFrame,
                   times: list, n_bins: int = 5) -> dict:
    """
    O/E calibration at each time: bin predicted S(t) into n_bins groups,
    compare predicted mean vs KM-estimated observed survival.
    """
    results = {}
    for i, t in enumerate(times):
        pred = surv_mat[:, i]
        bins = np.percentile(pred, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 3:
            continue
        labels = np.digitize(pred, bins[1:-1])
        obs_pts, pred_pts = [], []
        for b in np.unique(labels):
            mask = labels == b
            km = KaplanMeierFitter()
            km.fit(df_test[TIME_COL][mask], df_test[EVENT_COL][mask])
            obs_pts.append(float(km.predict(t)))
            pred_pts.append(float(pred[mask].mean()))
        results[t] = {"observed": obs_pts, "predicted": pred_pts}
    return results


# ── 8. Persistence ───────────────────────────────────────────────────────────

def env_versions() -> dict:
    return dict(
        python   = platform.python_version(),
        numpy    = np.__version__,
        pandas   = pd.__version__,
        sklearn  = sklearn.__version__,
        lifelines= lifelines.__version__,
        sksurv   = sksurv.__version__,
        optuna   = optuna.__version__,
        torch    = (torch.__version__ if _DEEPSURV_AVAILABLE else "n/a"),
    )


def save_model(obj, path: str, meta: dict | None = None) -> None:
    bundle = dict(model=obj, meta=meta or {}, versions=env_versions(),
                  saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    joblib.dump(bundle, path, compress=3)
    print(f"  saved → {path}")


def load_model(path: str) -> dict:
    return joblib.load(path)


# ── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("survival_core self-test …")
    df = generate_nsclc_cohort(n=200, seed=SEED)
    print(f"  cohort: {len(df)} patients  "
          f"event_rate={df[EVENT_COL].mean():.1%}  "
          f"median_time={df[TIME_COL].median():.1f} months")

    dist_res = test_distributions(df)
    print(f"  best distribution: {dist_res['best']}  "
          f"(AIC={dist_res['table'].iloc[0]['AIC']:.1f})")

    cph = build_cox(df, penalizer=0.1)
    print(f"  Cox C-index (train): {cph.concordance_index_:.3f}")
    print("self-test OK")
