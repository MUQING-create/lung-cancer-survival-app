"""
survival_app.py — TCGA-LUAD Lung Adenocarcinoma Overall Survival
================================================================
Cox Proportional Hazards  ×  Log-Logistic Accelerated Failure Time
Real clinical data: TCGA-LUAD · N = 509 patients · NCI GDC API
Cell Press visual style · Research & educational use only
"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import threading
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test as _mlr
from shiny import App, reactive, render, ui
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sksurv.metrics import brier_score as _brier_score
from sksurv.util import Surv

import survival_core as sc
from survival_core import (
    BRAND, EVENT_COL, SEED, TIME_COL,
    build_aft, build_cox, evaluate, test_distributions,
)

warnings.filterwarnings("ignore")

if os.name == "nt":
    _TMP_ROOT = Path(__file__).parent / ".cache" / "tmp"
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(_TMP_ROOT)

    class _SafeTemporaryDirectory:
        def __init__(self, suffix=None, prefix=None, dir=None,
                     ignore_cleanup_errors=True):
            self.name = tempfile.mkdtemp(
                suffix=suffix or "",
                prefix=prefix or "tmp",
                dir=dir or str(_TMP_ROOT),
            )

        def __enter__(self):
            return self.name

        def __exit__(self, exc_type, exc, tb):
            return False

        def cleanup(self):
            return None

    tempfile.TemporaryDirectory = _SafeTemporaryDirectory

sc.apply_cell_matplotlib_style(**{
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "lines.markersize": 3.6,
})

_COX_CLR = BRAND["navy"]
_AFT_CLR = BRAND["orange"]
_MUTED   = BRAND["brown"]

# ── 1. GDC download & preprocessing ──────────────────────────────────────────

_GDC_URL    = "https://api.gdc.cancer.gov/cases"
# On HF Spaces: /tmp is writable. Locally: fall back to a .cache/ sibling dir.
_HF_CACHE   = Path("/tmp/tcga_luad_gdc.json")
_LOCAL_CACHE = Path(__file__).parent / ".cache" / "tcga_luad_gdc.json"
_GDC_CACHE  = _HF_CACHE if _HF_CACHE.parent.exists() else _LOCAL_CACHE
_DAYS_PER_M = 365.25 / 12.0

_GDC_FIELDS = ",".join([
    "case_id", "demographic.vital_status", "demographic.days_to_death",
    "diagnoses.days_to_last_follow_up", "diagnoses.days_to_death",
    "diagnoses.age_at_diagnosis", "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t", "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
])

_STAGE_MAP = {
    "Stage I": 1, "Stage IA": 1, "Stage IA1": 1, "Stage IA2": 1,
    "Stage IA3": 1, "Stage IB": 1,
    "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2,
    "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
    "Stage IV": 4, "Stage IVA": 4, "Stage IVB": 4,
}
_T_MAP = {
    "T0": 0, "T1": 1, "T1a": 1, "T1b": 1, "T1c": 1, "T1mi": 1,
    "T2": 2, "T2a": 2, "T2b": 2, "T3": 3, "T4": 4,
}
_N_MAP = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}

FEAT_COLS = ["age", "stage", "t_stage", "n_stage", "m_stage"]
FEAT_DISPLAY = {
    "age":     "Age (years)",
    "stage":   "Stage (I–IV)",
    "t_stage": "T Stage (T1–T4)",
    "n_stage": "N Stage (N0–N3)",
    "m_stage": "M Stage (M0/M1)",
}


def _first_diag(diags: list, key: str):
    for d in (diags or []):
        v = d.get(key)
        if v is not None:
            return v
    return None


def _download_gdc() -> list:
    if _GDC_CACHE.exists():
        print(f"  GDC cache: {_GDC_CACHE}", flush=True)
        return json.loads(_GDC_CACHE.read_text())
    print("  Downloading TCGA-LUAD from NCI GDC ...", flush=True)
    filt = json.dumps({
        "op": "in",
        "content": {"field": "project.project_id", "value": ["TCGA-LUAD"]},
    })
    hits, from_idx = [], 0
    while True:
        r = requests.get(_GDC_URL, params={
            "filters": filt, "fields": _GDC_FIELDS,
            "size": "500", "from": str(from_idx), "format": "JSON",
        }, timeout=120, headers={"Accept": "application/json"})
        r.raise_for_status()
        page = r.json()["data"]
        hits.extend(page["hits"])
        from_idx += len(page["hits"])
        if from_idx >= page["pagination"]["total"]:
            break
    _GDC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _GDC_CACHE.write_text(json.dumps(hits))
    print(f"  Downloaded {len(hits)} patients", flush=True)
    return hits


def _preprocess(hits: list) -> pd.DataFrame:
    rows = []
    for h in hits:
        dem   = h.get("demographic") or {}
        diags = h.get("diagnoses") or []
        event = 1 if dem.get("vital_status", "").upper() == "DEAD" else 0
        t_days = (
            (dem.get("days_to_death") or _first_diag(diags, "days_to_death"))
            if event else _first_diag(diags, "days_to_last_follow_up")
        )
        age_d = _first_diag(diags, "age_at_diagnosis")
        m_raw = str(_first_diag(diags, "ajcc_pathologic_m") or "").upper()
        rows.append({
            TIME_COL:  float(t_days) / _DAYS_PER_M if t_days else None,
            EVENT_COL: event,
            "age":     float(age_d) / 365.25 if age_d else None,
            "stage":   _STAGE_MAP.get(_first_diag(diags, "ajcc_pathologic_stage") or ""),
            "t_stage": _T_MAP.get(_first_diag(diags, "ajcc_pathologic_t") or ""),
            "n_stage": _N_MAP.get(_first_diag(diags, "ajcc_pathologic_n") or ""),
            "m_stage": (1.0 if m_raw.startswith("M1") else
                        0.0 if m_raw.startswith("M0") else None),
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=[TIME_COL, EVENT_COL])
    return df[df[TIME_COL] > 0].reset_index(drop=True)


def _make_y(df: pd.DataFrame):
    return Surv.from_arrays(
        event=df[EVENT_COL].astype(bool).values,
        time=df[TIME_COL].values,
        name_event=EVENT_COL, name_time=TIME_COL,
    )


def _feat_matrix(df: pd.DataFrame,
                 imputer: SimpleImputer = None,
                 fit: bool = False) -> tuple[pd.DataFrame, SimpleImputer]:
    X = df[FEAT_COLS].copy().astype(float)
    if fit:
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X)
    return pd.DataFrame(imputer.transform(X), columns=FEAT_COLS, index=df.index), imputer


_BUNDLE_VERSION = 2


def _stage_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(i): int((df["stage"] == i).sum()) for i in range(1, 5)}


def _null_brier_curve(df_train: pd.DataFrame, y_train, y_test,
                      n_test: int, times: list | np.ndarray) -> dict:
    times_arr = np.array(times, dtype=float)
    if len(times_arr) == 0:
        return {"times": [], "values": []}
    try:
        km_ref = KaplanMeierFitter().fit(df_train[TIME_COL], df_train[EVENT_COL])
        ref_sf = np.array([km_ref.predict(t) for t in times_arr], dtype=float)
        null_m = np.tile(ref_sf, (n_test, 1))
        _, null_bs = _brier_score(y_train, y_test, null_m, times_arr)
        return {"times": times_arr.tolist(), "values": np.asarray(null_bs).tolist()}
    except Exception:
        return {"times": [], "values": []}


def _sanitized_bundle(raw: dict) -> tuple[dict, bool]:
    if raw.get("bundle_version") == _BUNDLE_VERSION:
        return raw, False

    df_train = raw.get("df_train")
    df_test = raw.get("df_test")
    df_all = raw.get("df_all")
    y_train = raw["y_train"]
    y_test = raw["y_test"]
    n_train = int(raw.get("n_train", len(y_train)))
    n_test = int(raw.get("n_test", len(y_test)))
    stage_source = df_all if isinstance(df_all, pd.DataFrame) else df_train
    stage_counts = raw.get("stage_counts")
    if stage_counts is None and isinstance(stage_source, pd.DataFrame):
        stage_counts = _stage_counts(stage_source)

    res_cox = raw["res_cox"]
    null_brier = raw.get("null_brier")
    if null_brier is None and isinstance(df_train, pd.DataFrame):
        null_brier = _null_brier_curve(
            df_train, y_train, y_test, n_test, res_cox.get("times", EVAL_TIMES)
        )

    keep = [
        "y_train", "y_test", "n_total", "ev_rate", "med_fu",
        "dist", "km_train", "cox", "aft", "res_cox", "res_aft",
        "tr_cox", "tr_aft", "imp_statistics", "feat_cols",
    ]
    clean = {k: raw[k] for k in keep if k in raw}
    clean.update(
        bundle_version=_BUNDLE_VERSION,
        n_train=n_train,
        n_test=n_test,
        stage_counts=stage_counts or {str(i): 0 for i in range(1, 5)},
        null_brier=null_brier or {"times": [], "values": []},
    )
    return clean, True


def _save_bundle(bundle: dict, path: Path) -> None:
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as _f:
        pickle.dump(bundle, _f)


# ── 2. Startup — load bundle if available, else download + train from scratch 

_BUNDLE_PATH = Path(__file__).parent / "tcga_luad_app_bundle.pkl"
EVAL_TIMES   = np.array([12., 24., 36., 48., 60.])


def _startup_from_bundle(path: Path) -> dict:
    import pickle
    print(f"  Loading bundle: {path}", flush=True)
    with open(path, "rb") as f:
        b = pickle.load(f)
    print(f"  N={b['n_total']}  events={b['ev_rate']:.0%}  "
          f"median FU={b['med_fu']:.0f}m", flush=True)
    return b


def _startup_from_scratch() -> dict:
    hits      = _download_gdc()
    df        = _preprocess(hits)
    n_total   = len(df)
    ev_rate   = df[EVENT_COL].mean()
    med_fu    = df[TIME_COL].median()
    print(f"  N={n_total}  events={ev_rate:.0%}  median FU={med_fu:.0f}m", flush=True)

    df_train, df_test = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df[EVENT_COL],
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    ft_train, imp = _feat_matrix(df_train, fit=True)
    ft_test,  _   = _feat_matrix(df_test,  imputer=imp)

    y_train = _make_y(df_train)
    y_test  = _make_y(df_test)

    print("  Distribution fitting ...", flush=True)
    dist     = test_distributions(df_train)
    km_train = KaplanMeierFitter().fit(
        df_train[TIME_COL], df_train[EVENT_COL], label="Kaplan-Meier"
    )

    print("  Training Cox PH ...", flush=True)
    cox = build_cox(df_train, feat_df=ft_train)
    print(f"    penalizer={cox._penalizer_used}  train C={cox.concordance_index_:.3f}",
          flush=True)

    print(f"  Training {dist['best']} AFT ...", flush=True)
    aft = build_aft(df_train, aft_class=dist["aft_class"], feat_df=ft_train)

    print("  Evaluating test set (n_boot=150) ...", flush=True)
    res_cox = evaluate(cox, "cox", df_test, y_train, y_test,
                       EVAL_TIMES, feat_df_test=ft_test, n_boot=150)
    res_aft = evaluate(aft, "aft", df_test, y_train, y_test,
                       EVAL_TIMES, feat_df_test=ft_test, n_boot=150)
    print(
        f"  Cox  C={res_cox['c_index']:.3f} [{res_cox['ci_lo']:.3f},{res_cox['ci_hi']:.3f}]"
        f"  IBS={res_cox['ibs']:.4f}", flush=True,
    )
    print(
        f"  AFT  C={res_aft['c_index']:.3f} [{res_aft['ci_lo']:.3f},{res_aft['ci_hi']:.3f}]"
        f"  IBS={res_aft['ibs']:.4f}", flush=True,
    )

    print("  Computing training-set metrics ...", flush=True)
    tr_cox = sc.evaluate_train(cox, "cox", df_train, y_train,
                               EVAL_TIMES, feat_df_train=ft_train)
    tr_aft = sc.evaluate_train(aft, "aft", df_train, y_train,
                               EVAL_TIMES, feat_df_train=ft_train)
    print(f"  Cox  train C={tr_cox['c_index']:.3f}  train IBS={tr_cox['ibs']:.4f}",
          flush=True)
    print(f"  AFT  train C={tr_aft['c_index']:.3f}  train IBS={tr_aft['ibs']:.4f}",
          flush=True)

    null_brier = _null_brier_curve(df_train, y_train, y_test, len(df_test),
                                   res_cox.get("times", EVAL_TIMES))

    return dict(
        bundle_version=_BUNDLE_VERSION,
        y_train=y_train, y_test=y_test,
        n_total=n_total, ev_rate=ev_rate, med_fu=med_fu,
        n_train=len(df_train), n_test=len(df_test),
        stage_counts=_stage_counts(df),
        dist=dist, km_train=km_train,
        cox=cox, aft=aft,
        res_cox=res_cox, res_aft=res_aft,
        tr_cox=tr_cox, tr_aft=tr_aft,
        null_brier=null_brier,
        imp_statistics=imp.statistics_.tolist(), feat_cols=FEAT_COLS,
    )


print("=" * 56, flush=True)
print("TCGA-LUAD Survival App — initialising", flush=True)
print("=" * 56, flush=True)

if _BUNDLE_PATH.exists():
    _B, _changed = _sanitized_bundle(_startup_from_bundle(_BUNDLE_PATH))
    if _changed:
        _save_bundle(_B, _BUNDLE_PATH)
        print(f"  Bundle sanitized: {_BUNDLE_PATH}", flush=True)
else:
    _B = _startup_from_scratch()
    _save_bundle(_B, _BUNDLE_PATH)
    print(f"  Bundle saved: {_BUNDLE_PATH}", flush=True)

# ── Unpack bundle ─────────────────────────────────────────────────────────────
_Y_TR     = _B["y_train"]
_Y_TE     = _B["y_test"]
# Rebuild imputer from saved statistics to avoid sklearn version mismatch
_IMP = SimpleImputer(strategy="median")
_IMP.fit(pd.DataFrame([_B["imp_statistics"]], columns=FEAT_COLS))
_IMP.statistics_ = np.array(_B["imp_statistics"])

N_TOTAL  = _B["n_total"]
EV_RATE  = _B["ev_rate"]
MED_FU   = _B["med_fu"]
N_TRAIN  = _B["n_train"]
N_TEST   = _B["n_test"]
STAGE_COUNTS = _B.get("stage_counts", {str(i): 0 for i in range(1, 5)})
NULL_BRIER = _B.get("null_brier", {"times": [], "values": []})

DIST     = _B["dist"]
KM_TRAIN = _B["km_train"]
COX      = _B["cox"]
AFT      = _B["aft"]
RES_COX  = _B["res_cox"]
RES_AFT  = _B["res_aft"]
TR_COX   = _B.get("tr_cox", {})
TR_AFT   = _B.get("tr_aft", {})

_best_name = ("Cox PH" if RES_COX["c_index"] >= RES_AFT["c_index"]
              else f"{DIST['best']} AFT")
_best_risk = RES_COX["risk"] if _best_name == "Cox PH" else RES_AFT["risk"]
print("  Ready.", flush=True)


# ── 3. Analytics: global visitor map ─────────────────────────────────────────
# Runtime integrations are optional. The app remains functional when these
# variables are absent, with visit logging and geo-enrichment disabled.
_IPINFO_TOKEN = os.environ.get("IPINFO_TOKEN", "").strip()
_SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://vmivonjwubhbvlufxkdd.supabase.co"
).strip()
# Supabase anon keys are public client credentials; table access is governed by RLS.
_SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtaXZvbmp3dWJoYnZsdWZ4a2RkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODMwODMyMDQsImV4cCI6MjA5ODY1OTIwNH0.IHH8dPFWYCkZ7eFhIx5zHY0QGMi1_pDM1ebBfzoHha0",
).strip()
_ANALYTICS_CONFIGURED = bool(
    _SUPABASE_URL and _SUPABASE_KEY
)
_ANALYTICS_STATE = {"error": None}
_APP_NAME_SV  = "nsclc-survival"

_COUNTRY_NAMES = {
    "AF":"Afghanistan","AL":"Albania","DZ":"Algeria","AR":"Argentina",
    "AU":"Australia","AT":"Austria","BD":"Bangladesh","BE":"Belgium",
    "BR":"Brazil","BG":"Bulgaria","CA":"Canada","CL":"Chile",
    "CN":"China","CO":"Colombia","HR":"Croatia","CZ":"Czech Republic",
    "DK":"Denmark","EG":"Egypt","FI":"Finland","FR":"France",
    "DE":"Germany","GH":"Ghana","GR":"Greece","HK":"Hong Kong",
    "HU":"Hungary","IN":"India","ID":"Indonesia","IR":"Iran",
    "IQ":"Iraq","IE":"Ireland","IL":"Israel","IT":"Italy",
    "JP":"Japan","JO":"Jordan","KZ":"Kazakhstan","KE":"Kenya",
    "KR":"South Korea","KW":"Kuwait","LB":"Lebanon","MY":"Malaysia",
    "MX":"Mexico","MA":"Morocco","NL":"Netherlands","NZ":"New Zealand",
    "NG":"Nigeria","NO":"Norway","PK":"Pakistan","PE":"Peru",
    "PH":"Philippines","PL":"Poland","PT":"Portugal","QA":"Qatar",
    "RO":"Romania","RU":"Russia","SA":"Saudi Arabia","SG":"Singapore",
    "ZA":"South Africa","ES":"Spain","SE":"Sweden","CH":"Switzerland",
    "TW":"Taiwan","TH":"Thailand","TN":"Tunisia","TR":"Turkey",
    "UA":"Ukraine","AE":"United Arab Emirates","GB":"United Kingdom",
    "US":"United States","VN":"Vietnam","YE":"Yemen","ZW":"Zimbabwe",
}


def _country_name(code: str) -> str:
    return _COUNTRY_NAMES.get((code or "").upper(), code or "")


def _lookup_ip_location(ip: str):
    if not ip or ip in ("127.0.0.1", "::1"):
        return None, None, None, None
    try:
        if _IPINFO_TOKEN:
            response = requests.get(
                f"https://ipinfo.io/{ip}/json",
                params={"token": _IPINFO_TOKEN},
                timeout=4,
            )
            if response.status_code != 200:
                return None, None, None, None
            data = response.json()
            loc = data.get("loc", "")
            lat, lon = map(float, loc.split(",")) if loc else (None, None)
            return data.get("country"), data.get("city"), lat, lon

        response = requests.get(f"https://ipwho.is/{ip}", timeout=4)
        data = response.json() if response.status_code == 200 else {}
        if not data.get("success"):
            return None, None, None, None
        return (
            data.get("country_code"),
            data.get("city"),
            data.get("latitude"),
            data.get("longitude"),
        )
    except Exception:
        return None, None, None, None


def _sb_headers_sv():
    return {
        "apikey":        _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }


def _log_visit_sv(country, city, lat, lon):
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        return
    if lat is None or lon is None:
        return
    try:
        requests.post(
            f"{_SUPABASE_URL}/rest/v1/visits",
            headers=_sb_headers_sv(),
            json={"app_name": _APP_NAME_SV, "country": country,
                  "city": city, "lat": lat, "lon": lon},
            timeout=5,
        )
    except Exception:
        pass


def _delete_null_visits_sv():
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        return
    try:
        hdrs = {k: v for k, v in _sb_headers_sv().items() if k != "Prefer"}
        requests.delete(
            f"{_SUPABASE_URL}/rest/v1/visits",
            headers=hdrs,
            params={"app_name": f"eq.{_APP_NAME_SV}", "lat": "is.null"},
            timeout=5,
        )
    except Exception:
        pass


def _fetch_visits_sv():
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        _ANALYTICS_STATE["error"] = "configuration"
        return []
    try:
        hdrs = {k: v for k, v in _sb_headers_sv().items() if k != "Prefer"}
        r = requests.get(
            f"{_SUPABASE_URL}/rest/v1/visits",
            headers=hdrs,
            params={"app_name": f"eq.{_APP_NAME_SV}",
                    "select": "country,city,lat,lon"},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            _ANALYTICS_STATE["error"] = None
            return data if isinstance(data, list) else []
        _ANALYTICS_STATE["error"] = f"http-{r.status_code}"
        return []
    except Exception:
        _ANALYTICS_STATE["error"] = "connection"
        return []


_WORLD_GEO_PATH_SV = Path(__file__).parent / "world.geojson"
_WORLD_GEO_SV = None


def _load_world_geo_sv():
    global _WORLD_GEO_SV
    if _WORLD_GEO_SV is None and _WORLD_GEO_PATH_SV.exists():
        with open(_WORLD_GEO_PATH_SV, encoding="utf-8") as _f:
            _WORLD_GEO_SV = json.load(_f)
    return _WORLD_GEO_SV


threading.Thread(target=_delete_null_visits_sv, daemon=True).start()


def _make_visit_map_sv(user_lat=None, user_lon=None):
    from collections import Counter
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    visits = _fetch_visits_sv()
    valid  = [v for v in visits if v.get("lat") is not None and v.get("lon") is not None]
    lats = [v["lat"] for v in valid]
    lons = [v["lon"] for v in valid]

    city_counts = Counter(
        (v.get("city") or "").strip()
        for v in valid
        if (v.get("city") or "").strip()
    )
    label_names = {city for city, _ in city_counts.most_common(4)}
    seen_cities: set[str] = set()
    city_labels = []
    for v in valid:
        city = (v.get("city") or "").strip()
        if city and city in label_names and city not in seen_cities:
            seen_cities.add(city)
            city_labels.append((v["lat"], v["lon"], city))

    fig, ax = plt.subplots(figsize=(7.0, 3.5), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 85)
    _cell_ax(fig, ax)

    geo = _load_world_geo_sv()
    if geo:
        patches = []
        for feat in geo.get("features", []):
            geom = feat.get("geometry") or {}
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            try:
                if gtype == "Polygon":
                    pts = np.array(coords[0])[:, :2]
                    patches.append(Polygon(pts, closed=True))
                elif gtype == "MultiPolygon":
                    for poly in coords:
                        pts = np.array(poly[0])[:, :2]
                        patches.append(Polygon(pts, closed=True))
            except Exception:
                pass
        if patches:
            pc = PatchCollection(
                patches, facecolor=BRAND["mint"], edgecolor=BRAND["edge"],
                linewidth=0.4, alpha=0.35, zorder=1,
            )
            ax.add_collection(pc)

    if lons:
        ax.scatter(lons, lats, s=28, color=BRAND["blue"], alpha=0.8,
                   marker="o", zorder=3, linewidths=0,
                   label=f"Visitors (n={len(lons)})")
        for i, (lat_c, lon_c, name) in enumerate(city_labels):
            if lon_c < -100:
                dx = 5
            elif lon_c > 105:
                dx = -4
            else:
                dx = 5 if i % 2 == 0 else -4
            dy = 4 if i % 3 == 0 else (-7 if i % 3 == 1 else 9)
            ax.annotate(name, xy=(lon_c, lat_c),
                        xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx >= 0 else "right",
                        fontsize=7.0, color=BRAND["ink"], zorder=5, clip_on=False)

    if user_lat is not None and user_lon is not None:
        ax.scatter([user_lon], [user_lat], s=32, color=BRAND["red"],
                   marker="o", zorder=4, linewidths=0, label="You")

    ax.set_xticks([])
    ax.set_yticks([])

    if lons or user_lat is not None:
        ax.legend(fontsize=8.0, loc="lower left", frameon=False)

    fig.tight_layout(pad=0.6)
    return fig

# ── 4. Plotting helpers ──────────────────────────────────────────────────────

_CURVE_T = np.linspace(0.1, 72, 300)
_KEY_T   = np.array([6., 12., 18., 24., 36., 48., 60.])
_REF_T   = np.array([12., 24., 36., 60.])


def _cell_ax(fig, ax):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(BRAND["edge"])
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=BRAND["ink"], direction="out", length=3, width=0.8, labelsize=8.0)
    ax.xaxis.label.set_color(BRAND["ink"])
    ax.yaxis.label.set_color(BRAND["ink"])
    ax.xaxis.label.set_size(9.0)
    ax.yaxis.label.set_size(9.0)
    ax.title.set_color(BRAND["ink"])
    ax.title.set_size(10.0)
    ax.grid(False)


def _predict_curve(model, feat_df: pd.DataFrame) -> np.ndarray:
    """Survival probabilities at _CURVE_T for one patient."""
    try:
        sf    = model.predict_survival_function(feat_df, times=_CURVE_T)
        probs = sf.iloc[:, 0].values
    except TypeError:
        sf    = model.predict_survival_function(feat_df)
        s     = sf.iloc[:, 0]
        probs = np.interp(_CURVE_T, s.index.values, s.values, left=1.0)
    return probs.clip(0.0, 1.0)


def _median_crossing(times: np.ndarray, probs: np.ndarray):
    below = np.where(probs <= 0.5)[0]
    if len(below) == 0:
        return None
    idx = below[0]
    if idx == 0:
        return float(times[0])
    x0, x1 = times[idx - 1], times[idx]
    y0, y1 = probs[idx - 1], probs[idx]
    if y0 == y1:
        return float(x1)
    return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))


def _survival_at_times(model, times: np.ndarray) -> np.ndarray:
    vals = model.survival_function_at_times(times)
    arr = np.asarray(vals, dtype=float).reshape(-1)
    return np.clip(arr, 0.0, 1.0)


def _survival_function_frame(model):
    sf = getattr(model, "survival_function_", None)
    if sf is None or sf.empty:
        return None
    vals = sf.iloc[:, 0].astype(float)
    return vals.index.to_numpy(dtype=float), vals.to_numpy(dtype=float)


# ── 4. CSS (Cell Press style) ────────────────────────────────────────────────

_CSS = """
:root {
  --red:#E64B35;--blue:#4DBBD5;--teal:#00A087;--navy:#3C5488;--salmon:#F39B7F;
  --lav:#8491B4;--mint:#91D1C2;--crimson:#DC0000;--brown:#7E6148;--tan:#B09C85;
  --ink:var(--navy);--accent:var(--blue);--accent-2:var(--teal);--surface:#FFFFFF;
  --line:rgba(60,84,136,.16);--muted:rgba(60,84,136,.78);--white:#FFFFFF;
  --shadow-sm:0 10px 28px rgba(60,84,136,.08);--shadow-lg:0 24px 60px rgba(60,84,136,.12);
  --r:18px; --r-sm:12px;
  --font:'Arial','Helvetica','Liberation Sans','DejaVu Sans',sans-serif;
}
html,body{
  height:100%;
  font-family:var(--font);
  font-size:15px;
  font-variant-numeric:tabular-nums;
  background:#F7FBFD;
  color:var(--ink);
  -webkit-font-smoothing:antialiased;
}

.navbar{
  background:rgba(255,255,255,.96)!important;border-bottom:1px solid var(--line)!important;
  padding:.88rem 1.8rem;box-shadow:var(--shadow-sm);position:relative;backdrop-filter:blur(14px);}
.navbar::after{content:"";position:absolute;left:0;right:0;bottom:-1px;height:4px;
  background:var(--accent);}
.navbar-brand{color:var(--ink)!important;font-weight:800;font-size:1.02rem;letter-spacing:.18px;}

.bslib-sidebar-layout>.sidebar{
  background:rgba(255,255,255,.95)!important;border-right:1px solid var(--line)!important;
  box-shadow:var(--shadow-sm);overflow-y:auto;height:100%;padding:1.1rem 1.2rem 1.5rem;}

.sec{font-size:.74rem;font-weight:800;color:var(--ink);text-transform:uppercase;
  letter-spacing:1.6px;margin:1.4rem 0 .8rem;padding:0 0 .35rem .8rem;
  border-left:5px solid var(--accent-2);line-height:1.35;}
.sec:first-child{margin-top:.25rem;}

.form-label{font-size:.95rem;font-weight:700;color:var(--ink);margin-bottom:.42rem;display:block;}
.form-control,.form-select{font-size:.86rem;border:1px solid rgba(60,84,136,.18);border-radius:var(--r-sm);
  background:var(--white);padding:.7rem .86rem;min-height:2.95rem;color:var(--ink);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.7);
  transition:border-color .16s,box-shadow .16s,background .16s,transform .16s;}
.form-control:focus,.form-select:focus{border-color:var(--accent);background:white;
  box-shadow:0 0 0 4px rgba(77,187,213,.18);outline:none;}
.form-select{
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%2300A087' stroke-width='1.8' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 12px center;background-size:12px;
  padding-right:2.3rem;-webkit-appearance:none;appearance:none;cursor:pointer;}
.btn-primary{
  background:var(--accent)!important;
  border:none!important;border-radius:14px!important;font-size:.88rem!important;
  font-weight:700!important;letter-spacing:.1px;padding:.84rem 1rem!important;
  box-shadow:0 12px 24px rgba(60,84,136,.16);}
.btn-primary:hover{transform:translateY(-1px);}

.bslib-sidebar-layout>.main{padding:clamp(18px,2.6vw,30px)!important;}
.card-body{padding:clamp(14px,1.7vw,20px)!important;}

.page-title{font-size:clamp(1.55rem,1.9vw,2rem);font-weight:800;color:var(--ink);
  display:inline-block;margin:.15rem 0 .3rem;letter-spacing:-.03em;position:relative;}
.page-title::after{content:"";display:block;height:4px;width:min(100%,26rem);
  background:var(--accent);border-radius:999px;margin-top:.6rem;}
.page-subtitle{color:var(--muted);font-size:.84rem;margin-bottom:1.25rem;line-height:1.58;max-width:70rem;}

.infobar{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:18px;}
.metric-chip{background:rgba(255,255,255,.92);border:1px solid var(--line);border-top:4px solid var(--accent);
  border-radius:16px;padding:.88rem 1rem;display:flex;flex-direction:column;gap:.25rem;
  box-shadow:var(--shadow-sm);}
.metric-chip:nth-child(2n){border-top-color:var(--teal);}
.metric-chip:nth-child(3n){border-top-color:var(--navy);}
.metric-chip:nth-child(4n){border-top-color:var(--salmon);}
.mc-label{font-size:.62rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.9px;}
.mc-value{font-size:1rem;font-weight:800;color:var(--ink);white-space:nowrap;}
.mc-cox{color:var(--ink)!important;}
.mc-aft{color:var(--accent-2)!important;}

.card{border:1px solid var(--line)!important;border-radius:var(--r)!important;
  box-shadow:var(--shadow-sm)!important;background:rgba(255,255,255,.93)!important;
  overflow:hidden;margin-bottom:16px;position:relative;}
.card::before{content:"";display:block;height:4px;
  background:var(--accent);}
.card-header{background:rgba(255,255,255,.88)!important;border-bottom:1px solid rgba(60,84,136,.10)!important;
  color:var(--ink)!important;font-weight:800;font-size:.88rem;letter-spacing:.14px;padding:1rem 1.2rem;}
.plot-frame{width:100%;height:clamp(300px,32vw,420px);
  display:flex;align-items:center;justify-content:center;overflow:hidden;}
.plot-frame.plot-map{height:clamp(260px,31vw,390px);}
.plot-frame.plot-survival{height:clamp(320px,30vw,400px);}
.plot-frame.plot-tall{height:clamp(360px,31vw,460px);}
.plot-frame .shiny-plot-output{width:100%!important;height:100%!important;}
.plot-frame .shiny-plot-output img,.plot-frame .shiny-plot-output canvas{
  width:100%!important;height:100%!important;max-width:100%!important;
  max-height:100%!important;object-fit:contain!important;object-position:center center!important;}
.equal-card{height:100%;display:flex;flex-direction:column;}
.equal-card .card-body{flex:1 1 auto;min-height:0;display:flex;flex-direction:column;}
.plot-frame,.result-frame{flex:1 1 auto;min-height:0;}
.result-frame{display:flex;flex-direction:column;justify-content:flex-start;}
.result-frame.result-map{min-height:clamp(260px,31vw,390px);}
.result-frame.result-survival,.result-frame.result-dist{min-height:clamp(320px,34vw,440px);}
.nav-tabs{border:0!important;margin-bottom:18px;gap:10px;flex-wrap:wrap;}
.nav-tabs .nav-link{color:var(--muted)!important;font-size:.84rem;font-weight:700;
  border:1px solid rgba(60,84,136,.14)!important;padding:.68rem 1.1rem;
  margin-bottom:0;border-radius:999px;background:rgba(255,255,255,.82)!important;
  box-shadow:0 6px 18px rgba(60,84,136,.05);
  transition:color .14s,background .14s,transform .14s,box-shadow .14s,border-color .14s;}
.nav-tabs .nav-link:hover{color:var(--ink)!important;background:rgba(77,187,213,.10)!important;
  border-color:rgba(77,187,213,.32)!important;transform:translateY(-1px);}
.nav-tabs .nav-link.active{color:var(--ink)!important;font-weight:800;
  border-color:transparent!important;background:rgba(77,187,213,.18)!important;
  box-shadow:var(--shadow-sm);}

.prob-tbl{width:100%;border-collapse:collapse;font-size:.82rem;}
.prob-tbl th{text-align:left;color:var(--ink);font-weight:700;padding:.72rem .8rem;
  border-bottom:2px solid rgba(60,84,136,.12);font-size:.64rem!important;text-transform:uppercase;
  letter-spacing:.65px;white-space:nowrap;}
.prob-tbl td{padding:.72rem .8rem;border-bottom:1px solid rgba(132,145,180,.14);
  font-variant-numeric:tabular-nums;font-size:.82rem!important;}
.prob-tbl td.val{font-weight:700;}
.prob-tbl td.cox{color:var(--ink);}
.prob-tbl td.aft{color:var(--accent-2);}
.prob-tbl tr:hover td{background:rgba(145,209,194,.12);}
.prob-card{display:flex;flex-direction:column;gap:10px;margin-top:2px;}
.prob-row{display:grid;grid-template-columns:56px minmax(0,1fr) 64px;gap:10px;align-items:center;
  padding:9px 0;border-bottom:1px solid rgba(132,145,180,.14);}
.prob-time{font-size:.72rem;font-weight:800;color:var(--ink);}
.prob-track{display:grid;gap:5px;min-width:0;}
.prob-bar{height:8px;background:rgba(132,145,180,.14);border-radius:999px;overflow:hidden;}
.prob-fill{height:100%;border-radius:999px;}
.prob-meta{display:flex;justify-content:space-between;gap:8px;font-size:.67rem;color:var(--muted);
  white-space:nowrap;}
.prob-diff{text-align:right;font-size:.72rem;font-weight:800;}

.methods{font-size:.82rem;line-height:1.58;color:var(--ink);}
.methods h4{font-size:.78rem;font-weight:800;color:var(--accent);
  text-transform:uppercase;letter-spacing:1.2px;margin:16px 0 8px;
  padding-left:10px;border-left:4px solid var(--accent-2);}
.methods p{margin:0 0 12px;}
.mtbl{width:100%;border-collapse:collapse;font-size:.82rem;}
.mtbl th{text-align:left;color:var(--ink);font-weight:700;padding:.72rem .7rem;
  border-bottom:2px solid rgba(60,84,136,.12);font-size:.64rem!important;text-transform:uppercase;
  letter-spacing:.65px;}
.mtbl td{padding:.72rem .7rem;border-bottom:1px solid rgba(132,145,180,.14);font-size:.82rem!important;}
.mtbl td.num{text-align:right;font-variant-numeric:tabular-nums;font-weight:700;color:var(--ink);}
.mtbl tr.best td{background:rgba(145,209,194,.16);}

.disclaimer{color:var(--muted);font-size:.74rem;margin-top:10px;padding-top:10px;
  border-top:1px solid var(--line);text-align:center;line-height:1.65;}

.bslib-page-fill{height:100dvh!important;}
.tab-content{flex:1 1 auto;min-height:0;display:flex;flex-direction:column;}
.tab-pane.active{flex:1 1 auto;min-height:0;display:flex!important;flex-direction:column;gap:0;}
.tab-pane.active>*{flex-shrink:0;}
@media (max-width: 900px){
  html,body{font-size:14px;}
  .bslib-sidebar-layout>.sidebar{padding:1rem .95rem 1.3rem;}
  .bslib-sidebar-layout>.main{padding:14px!important;}
  .page-title{font-size:clamp(1.4rem,6.5vw,1.8rem);}
  .page-subtitle{font-size:.8rem;}
  .plot-frame{height:clamp(270px,72vw,380px);}
  .plot-frame.plot-map{height:clamp(220px,64vw,320px);}
  .plot-frame.plot-survival{height:clamp(300px,74vw,400px);}
  .plot-frame.plot-tall{height:clamp(360px,82vw,450px);}
  .result-frame.result-map{min-height:clamp(220px,64vw,320px);}
  .result-frame.result-survival,.result-frame.result-dist{min-height:clamp(260px,74vw,380px);}
  .nav-tabs .nav-link{padding:.5rem .82rem;font-size:.8rem;}
  .prob-row{grid-template-columns:48px minmax(0,1fr) 56px;gap:8px;}
}
"""

_CSS += """
:root{--r:8px;--r-sm:8px;}
.card,.metric-chip{border-radius:8px!important;}
.btn-primary{border-radius:8px!important;}
.hero-copy{margin-bottom:1rem;}
.hero-kicker{
  font-size:.68rem;
  font-weight:800;
  color:var(--accent-2);
  text-transform:uppercase;
  letter-spacing:1.1px;
  margin-bottom:.35rem;
}
.summary-grid{
  display:grid;
  grid-template-columns:repeat(4,minmax(0,1fr));
  gap:12px;
  margin:0 0 18px;
}
.summary-tile{
  position:relative;
  overflow:hidden;
  background:rgba(255,255,255,.9);
  border:1px solid var(--line);
  border-radius:8px;
  padding:14px 14px 12px;
  box-shadow:0 8px 22px rgba(60,84,136,.08);
}
.summary-tile::before{
  content:"";
  position:absolute;
  left:0;
  top:0;
  bottom:0;
  width:4px;
  background:var(--tile);
}
.summary-tile.accent-blue{--tile:var(--blue);}
.summary-tile.accent-teal{--tile:var(--teal);}
.summary-tile.accent-navy{--tile:var(--navy);}
.summary-tile.accent-salmon{--tile:var(--salmon);}
.summary-label{
  font-size:.66rem;
  font-weight:800;
  color:var(--muted);
  text-transform:uppercase;
  letter-spacing:.8px;
}
.summary-value{
  font-size:1.24rem;
  font-weight:800;
  color:var(--ink);
  line-height:1.1;
  margin-top:8px;
}
.summary-detail{
  font-size:.78rem;
  color:var(--muted);
  line-height:1.48;
  margin-top:7px;
}
.section-head{
  display:flex;
  flex-direction:column;
  gap:4px;
  margin:0 0 14px;
}
.section-eyebrow{
  font-size:.66rem;
  font-weight:800;
  color:var(--accent-2);
  text-transform:uppercase;
  letter-spacing:.9px;
}
.section-title{
  margin:0;
  font-size:1.08rem;
  font-weight:800;
  color:var(--ink);
}
.section-copy{
  margin:0;
  max-width:60rem;
  font-size:.82rem;
  line-height:1.55;
  color:var(--muted);
}
.stage-grid{
  display:grid;
  grid-template-columns:repeat(4,minmax(0,1fr));
  gap:12px;
  margin:0 0 18px;
}
.stage-tile{
  position:relative;
  overflow:hidden;
  background:rgba(255,255,255,.88);
  border:1px solid var(--line);
  border-radius:8px;
  padding:14px 14px 12px;
  box-shadow:0 8px 22px rgba(60,84,136,.08);
}
.stage-tile::before{
  content:"";
  position:absolute;
  inset:0 auto 0 0;
  width:4px;
  background:var(--stage);
}
.stage-1{--stage:var(--blue);}
.stage-2{--stage:var(--teal);}
.stage-3{--stage:var(--navy);}
.stage-4{--stage:var(--salmon);}
.stage-kicker{
  font-size:.68rem;
  font-weight:800;
  color:var(--muted);
  text-transform:uppercase;
  letter-spacing:.8px;
}
.stage-value{
  font-size:1.2rem;
  font-weight:800;
  color:var(--ink);
  line-height:1.1;
  margin-top:8px;
}
.stage-detail{
  font-size:.78rem;
  color:var(--muted);
  line-height:1.48;
  margin-top:7px;
}
.note-grid{
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:12px;
  margin:0 0 18px;
}
.note-block{
  background:rgba(255,255,255,.72);
  border:1px dashed rgba(60,84,136,.28);
  border-radius:8px;
  padding:12px 14px;
}
.note-title{
  font-size:.72rem;
  font-weight:800;
  color:var(--ink);
  text-transform:uppercase;
  letter-spacing:.65px;
}
.note-copy{
  margin:6px 0 0;
  font-size:.78rem;
  line-height:1.5;
  color:var(--muted);
}
@media (max-width: 1100px){
  .summary-grid,.stage-grid,.note-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
}
@media (max-width: 700px){
  .summary-grid,.stage-grid,.note-grid{grid-template-columns:1fr;}
}
"""

_CSS += """
*{letter-spacing:0!important;}
html,body{background:#FFFFFF;}
.navbar{
  background:#FFFFFF!important;
  box-shadow:none;
  backdrop-filter:none;
  padding:.78rem 1.35rem;
}
.navbar::after{height:3px;background:var(--red);}
.bslib-sidebar-layout>.sidebar{
  background:#FFFFFF!important;
  box-shadow:none;
}
.bslib-sidebar-layout>.main{background:#FFFFFF;}
.page-title{font-size:1.7rem;letter-spacing:0;}
.page-title::after{height:3px;border-radius:0;background:var(--red);}
.card,.summary-tile,.stage-tile,.metric-chip{
  background:#FFFFFF!important;
  border:1px solid rgba(60,84,136,.28)!important;
  box-shadow:none!important;
  border-radius:8px!important;
}
.card::before{height:3px;background:var(--blue);}
.card-header{background:#FFFFFF!important;padding:.82rem 1rem;}
.summary-tile,.stage-tile{border-top:3px solid var(--tile,var(--stage))!important;}
.summary-tile::before,.stage-tile::before{display:none;}
.note-block{
  background:#FFFFFF;
  border:0;
  border-left:3px solid var(--mint);
  border-radius:0;
  padding:9px 12px;
}
.nav-tabs .nav-link{box-shadow:none;}
.btn-primary{box-shadow:none!important;}
.responsive-plot-mobile{display:none;}
.responsive-figure img{display:block;width:100%;height:auto;}
@media (max-width:700px){
  .page-title{font-size:1.45rem;}
  .navbar{padding:.68rem .85rem;}
  .responsive-plot-desktop{display:none;}
  .responsive-plot-mobile{display:flex;}
}
"""


# ── 5. UI ────────────────────────────────────────────────────────────────────

_stage_lbl  = {"1": "I",  "2": "II",  "3": "III",  "4": "IV"}
_t_lbl      = {"1": "T1", "2": "T2",  "3": "T3",   "4": "T4"}
_n_lbl      = {"0": "N0", "1": "N1",  "2": "N2",   "3": "N3"}
_m_lbl      = {"0": "M0 — No distant metastasis", "1": "M1 — Distant metastasis"}


def _summary_tile(label: str, value: str, detail: str, accent: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(label, class_="summary-label"),
        ui.tags.div(value, class_="summary-value"),
        ui.tags.div(detail, class_="summary-detail"),
        class_=f"summary-tile {accent}",
    )


def _section_head(kicker: str, title: str, copy: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(kicker, class_="section-eyebrow"),
        ui.tags.h4(title, class_="section-title"),
        ui.tags.p(copy, class_="section-copy"),
        class_="section-head",
    )


def _stage_tile(stage_code: str) -> ui.Tag:
    count = int(STAGE_COUNTS.get(stage_code, 0))
    share = (100.0 * count / N_TOTAL) if N_TOTAL else 0.0
    return ui.tags.div(
        ui.tags.div(f"Stage {_stage_lbl[stage_code]}", class_="stage-kicker"),
        ui.tags.div(f"{count}", class_="stage-value"),
        ui.tags.div(f"{share:.0f}% of cohort", class_="stage-detail"),
        class_=f"stage-tile stage-{stage_code}",
    )


def _note_block(title: str, copy: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(title, class_="note-title"),
        ui.tags.p(copy, class_="note-copy"),
        class_="note-block",
    )

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.div("Patient Profile", class_="sec"),

        ui.input_numeric("age", "Age (years)", value=65, min=18, max=100, step=1),

        ui.input_select(
            "stage", "AJCC Stage",
            {k: f"Stage {v}" for k, v in _stage_lbl.items()},
            selected="2",
        ),
        ui.input_select("t_stage", "T Stage", _t_lbl, selected="2"),
        ui.input_select("n_stage", "N Stage", _n_lbl, selected="0"),
        ui.input_select("m_stage", "M Stage", _m_lbl, selected="0"),

        ui.input_action_button(
            "submit", "Generate Forecast",
            class_="btn btn-primary w-100",
            style="margin-top:10px;font-weight:600;",
        ),
        ui.tags.div("Model Stack", class_="sec"),
        ui.tags.div(
            ui.tags.span(
                "■ Cox PH",
                style=f"color:{_COX_CLR};font-weight:600;font-size:.80rem;",
            ),
            ui.tags.span(
                " ■ Log-Logistic AFT",
                style=f"color:{_AFT_CLR};font-weight:600;font-size:.80rem;",
            ),
            style="line-height:2;",
        ),
        ui.tags.p(
            f"TCGA-LUAD · train {N_TRAIN} / test {N_TEST} · median-imputed inputs",
            style=f"font-size:.72rem;color:{_MUTED};margin:4px 0 0;",
        ),

        width=300,
    ),

    ui.tags.style(_CSS),
    ui.tags.div(
        ui.tags.div("Survival atlas", class_="hero-kicker"),
        ui.tags.h3(
            "TCGA-LUAD Survival Forecast Atlas",
            class_="page-title",
        ),
        ui.tags.p(
            f"Real TCGA-LUAD cohort from the NCI GDC API · N={N_TOTAL} patients · "
            f"event rate {EV_RATE:.0%} · median follow-up {MED_FU:.0f} months. "
            f"The app contrasts Cox PH with {DIST['best']} AFT using a fixed train/test split.",
            class_="page-subtitle",
        ),
        class_="hero-copy",
    ),
    ui.tags.div(
        _summary_tile(
            "Cohort",
            f"{N_TOTAL}",
            f"train {N_TRAIN} / test {N_TEST} patients",
            "accent-blue",
        ),
        _summary_tile(
            "Event burden",
            f"{EV_RATE:.0%}",
            "overall survival event rate in the full cohort",
            "accent-teal",
        ),
        _summary_tile(
            "Follow-up",
            f"{MED_FU:.0f} m",
            "median observed follow-up time",
            "accent-navy",
        ),
        _summary_tile(
            "Best test C-index",
            f"{max(RES_COX['c_index'], RES_AFT['c_index']):.3f}",
            f"{_best_name} on held-out patients",
            "accent-salmon",
        ),
        class_="summary-grid",
    ),

    ui.navset_tab(
        ui.nav_panel(
            "Patient Explorer",
            _section_head(
                "Forecast desk",
                "Patient-specific survival projection",
                "The first view focuses on one patient at a time: two survival models, key time-point probabilities, and compact interpretation cues.",
            ),
            ui.output_ui("info_bar"),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Predicted Survival Curves"),
                    ui.tags.div(
                        ui.output_plot("survival_curve", width="100%", height="100%"),
                        class_="plot-frame plot-survival",
                    ),
                    class_="equal-card",
                ),
                ui.card(
                    ui.card_header("Survival Probability at Key Time Points"),
                    ui.tags.div(ui.output_ui("prob_table"), class_="result-frame result-survival"),
                    class_="equal-card",
                ),
                col_widths=[7, 5],
            ),
            ui.tags.div(
                _note_block(
                    "Inputs used",
                    "Age plus AJCC stage, T stage, N stage, and M stage drive both models. Missing values are handled with training-set median imputation.",
                ),
                _note_block(
                    "Model pairing",
                    f"Cox PH and {DIST['best']} AFT are shown together so agreement and divergence remain visible instead of hidden behind a single score.",
                ),
                _note_block(
                    "Use boundary",
                    "This interface is for research and educational review. It exposes train/test behavior and is not a clinical decision system.",
                ),
                class_="note-grid",
            ),
            ui.tags.p(
                "Based on real TCGA-LUAD data. For research and educational use only. "
                "Not a clinical diagnostic tool.",
                class_="disclaimer",
            ),
        ),

        ui.nav_panel(
            "Cohort Landscape",
            _section_head(
                "Cohort map",
                "Stage distribution, survival shape, and usage footprint",
                "This view borrows a landscape structure: stage composition first, then the marginal survival fit, then the global audience footprint.",
            ),
            ui.tags.div(
                _stage_tile("1"),
                _stage_tile("2"),
                _stage_tile("3"),
                _stage_tile("4"),
                class_="stage-grid",
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header(
                        "Marginal Survival Distribution — Training Data"
                    ),
                    ui.tags.div(
                        ui.output_plot("dist_plot", width="100%", height="100%"),
                        class_="plot-frame plot-survival",
                    ),
                    class_="equal-card",
                ),
                ui.card(
                    ui.card_header("AIC / BIC Comparison"),
                    ui.tags.div(
                        ui.output_ui("dist_table"),
                        ui.tags.div(
                            ui.tags.h4(
                                "Why parametric distribution selection matters",
                                style=f"font-size:.64rem;font-weight:700;color:{_MUTED};"
                                      "text-transform:uppercase;letter-spacing:.6px;"
                                      "margin:12px 0 5px;",
                            ),
                            ui.tags.p(
                                "The Akaike Information Criterion (AIC) penalises model "
                                "complexity while rewarding goodness-of-fit. "
                                "The distribution with the lowest AIC best captures the "
                                "marginal shape of the observed survival times.",
                                style="font-size:.70rem;line-height:1.48;color:var(--ct);",
                            ),
                            ui.tags.p(
                                f"Best fit: ",
                                ui.tags.strong(DIST["best"]),
                                f" — selected as the AFT family for the parametric model. "
                                "Unlike Cox PH, this parametric form supports analytical "
                                "extrapolation beyond the 72-month follow-up window.",
                                style="font-size:.70rem;line-height:1.48;color:var(--ct);",
                            ),
                            style="padding:2px 2px 0;",
                        ),
                        class_="result-frame result-dist",
                    ),
                    class_="equal-card",
                ),
                col_widths=[7, 5],
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Global Visitor Map"),
                    ui.tags.div(
                        ui.output_plot("visit_map", width="100%", height="100%"),
                        class_="plot-frame plot-map",
                    ),
                    class_="equal-card",
                ),
                ui.card(
                    ui.card_header("Visit Statistics"),
                    ui.tags.div(ui.output_ui("visit_stats"), class_="result-frame result-map"),
                    class_="equal-card",
                ),
                col_widths=[8, 4],
            ),
        ),

        ui.nav_panel(
            "Model Arena",
            _section_head(
                "Benchmark view",
                "Train and test performance in one place",
                "The arena view keeps rank ordering, uncertainty, and overfitting signals together so model choice stays evidence-driven.",
            ),
            ui.output_ui("perf_chips"),
            ui.tags.div(
                _summary_tile(
                    "AFT family",
                    DIST["best"],
                    f"AIC-selected from {len(DIST['table'])} parametric candidates",
                    "accent-teal",
                ),
                _summary_tile(
                    "Cox test C-index",
                    f"{RES_COX['c_index']:.3f}",
                    f"train {TR_COX.get('c_index', float('nan')):.3f} / test {RES_COX['c_index']:.3f}",
                    "accent-blue",
                ),
                _summary_tile(
                    "AFT test C-index",
                    f"{RES_AFT['c_index']:.3f}",
                    f"train {TR_AFT.get('c_index', float('nan')):.3f} / test {RES_AFT['c_index']:.3f}",
                    "accent-navy",
                ),
                _summary_tile(
                    "Evaluation window",
                    "12-60 m",
                    "IBS and dynamic AUC are aligned to shared follow-up times",
                    "accent-salmon",
                ),
                class_="summary-grid",
            ),
            ui.card(
                ui.card_header(
                    f"Train / Test Comparison — N={N_TEST} held-out patients"
                ),
                ui.tags.div(
                    ui.output_ui("perf_plot_desktop"),
                    class_="responsive-figure responsive-plot-desktop",
                ),
                ui.tags.div(
                    ui.output_ui("perf_plot_mobile"),
                    class_="responsive-figure responsive-plot-mobile",
                ),
            ),
            ui.tags.div(
                _note_block(
                    "Bootstrap uncertainty",
                    "Test-set C-index intervals are estimated with 150 bootstrap resamples so rank differences are not read as exact.",
                ),
                _note_block(
                    "IBS horizon",
                    "Integrated Brier Score is evaluated at 12, 24, 36, 48, and 60 months after clipping to the observed support.",
                ),
                _note_block(
                    "Overfitting check",
                    "The train/test pairing is deliberately visible. A large gap is the first warning sign that a model is memorizing the training cohort.",
                ),
                class_="note-grid",
            ),
        ),

        ui.nav_panel(
            "Methods Atlas",
            _section_head(
                "Reproducibility",
                "Bundle contents, preprocessing, and evaluation rules",
                "Everything below is tied to the deployed bundle so the modeling assumptions behind the interface remain inspectable.",
            ),
            ui.output_ui("methods_panel"),
        ),

        id="main_tab",
    ),

    title="TCGA-LUAD Survival Atlas",
    fillable=True,
)


# ── 6. Server ────────────────────────────────────────────────────────────────

def server(input, output, session):

    # ── Visit logging ─────────────────────────────────────────────────────────
    _user_loc = {"lat": None, "lon": None}
    try:
        _hdrs = session.http_conn.headers
        _ip = (
            _hdrs.get("x-forwarded-for") or
            _hdrs.get("x-real-ip") or ""
        ).split(",")[0].strip()
    except Exception:
        _ip = ""

    def _do_log(ip: str) -> None:
        country, city, lat, lon = _lookup_ip_location(ip)
        if lat is not None and lon is not None:
            _user_loc["lat"] = lat
            _user_loc["lon"] = lon
        _log_visit_sv(country, city, lat, lon)

    threading.Thread(target=_do_log, args=(_ip,), daemon=True).start()

    # Auto-refresh the map/stats a few times after load so the visitor's own
    # just-logged visit (written asynchronously above) appears without a
    # manual page refresh. Bounded to a handful of ticks, then stops.
    _refresh_tick = reactive.value(0)
    _refresh_n    = {"c": 0}

    @reactive.effect
    def _auto_refresh():
        _refresh_n["c"] += 1
        if _refresh_n["c"] <= 3:
            reactive.invalidate_later(3)
            _refresh_tick.set(_refresh_n["c"])

    @render.plot
    def visit_map():
        _refresh_tick.get()  # re-render when the tick advances
        return _make_visit_map_sv(_user_loc["lat"], _user_loc["lon"])

    @render.ui
    def visit_stats():
        from collections import Counter
        _refresh_tick.get()  # re-render when the tick advances
        visits = _fetch_visits_sv()
        total  = len(visits)
        counts = Counter(
            f"{v.get('city')}, {_country_name(v.get('country'))}" if v.get("country") else v.get("city")
            for v in visits if v.get("city")
        )
        top    = counts.most_common(10)
        rows   = "".join(
            f"<tr><td style='font-size:.70rem;'>{i+1}. {c}</td>"
            f"<td class='num' style='font-size:.70rem;'>{n}</td></tr>"
            for i, (c, n) in enumerate(top)
        )
        empty_message = (
            "Visit analytics are temporarily unavailable."
            if _ANALYTICS_STATE["error"]
            else "No visits recorded yet."
        )
        empty = (
            f'<p style="font-size:.64rem;color:{_MUTED};padding:8px 0;">'
            f"{empty_message}</p>"
            if not top else ""
        )
        return ui.HTML(f"""
<div style="padding:4px 6px;">
  <div style="text-align:center;margin-bottom:12px;">
    <div style="font-size:1.8rem;font-weight:700;color:{_COX_CLR};">{total}</div>
    <div style="font-size:.62rem;color:{_MUTED};text-transform:uppercase;
                letter-spacing:.8px;">Total Visits</div>
  </div>
  <table class="prob-tbl" style="width:100%;">
    <thead><tr>
      <th>City</th>
      <th style="text-align:right;">Visits</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {empty}
</div>
""")

    # ── Patient feature dataframe ─────────────────────────────────────────────

    @reactive.calc
    def patient_feat() -> pd.DataFrame:
        input.submit()
        with reactive.isolate():
            row = {
                "age":     float(input.age()),
                "stage":   float(input.stage()),
                "t_stage": float(input.t_stage()),
                "n_stage": float(input.n_stage()),
                "m_stage": float(input.m_stage()),
            }
        df_pt = pd.DataFrame([row])
        feat, _ = _feat_matrix(df_pt, imputer=_IMP)
        return feat

    @reactive.calc
    def curves():
        feat   = patient_feat()
        s_cox  = _predict_curve(COX, feat)
        s_aft  = _predict_curve(AFT, feat)
        return s_cox, s_aft

    # ── Info bar ──────────────────────────────────────────────────────────────

    @render.ui
    def info_bar():
        s_cox, s_aft = curves()
        chips = []
        for label, t, sc, sa in [
            ("1-yr OS", 12.0,  np.interp(12.0,  _CURVE_T, s_cox),
                                np.interp(12.0,  _CURVE_T, s_aft)),
            ("3-yr OS", 36.0,  np.interp(36.0,  _CURVE_T, s_cox),
                                np.interp(36.0,  _CURVE_T, s_aft)),
            ("5-yr OS", 60.0,  np.interp(60.0,  _CURVE_T, s_cox),
                                np.interp(60.0,  _CURVE_T, s_aft)),
        ]:
            chips.append(
                f'<div class="metric-chip">'
                f'<span class="mc-label">{label} · Cox PH</span>'
                f'<span class="mc-value mc-cox">{sc*100:.1f}%</span>'
                f'</div>'
                f'<div class="metric-chip">'
                f'<span class="mc-label">{label} · AFT</span>'
                f'<span class="mc-value mc-aft">{sa*100:.1f}%</span>'
                f'</div>'
            )
        return ui.HTML('<div class="infobar">' + "".join(chips) + "</div>")

    # ── Survival curve plot ───────────────────────────────────────────────────

    @render.plot
    def survival_curve():
        s_cox, s_aft = curves()

        fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=300)
        _cell_ax(fig, ax)

        ax.plot(_CURVE_T, s_cox, color=_COX_CLR, lw=1.0, label="Cox PH", zorder=4)
        ax.plot(_CURVE_T, s_aft, color=_AFT_CLR, lw=1.0, ls="--",
                label=f"{DIST['best']} AFT", zorder=4)

        ax.fill_between(_CURVE_T, s_cox, s_aft,
                        alpha=0.10, color=BRAND["mint"], linewidth=0, zorder=1)

        for t_ref in _REF_T:
            ax.axvline(t_ref, color=BRAND["edge"], lw=0.8, ls=":", zorder=0)

        ax.axhline(0.5, color=BRAND["edge"], lw=1.0, ls="--", zorder=0)
        ax.text(71.5, 0.515, "50%", ha="right", va="bottom",
                fontsize=7.0, color=_MUTED)

        for t_ann in _REF_T:
            sc = float(np.interp(t_ann, _CURVE_T, s_cox))
            sa = float(np.interp(t_ann, _CURVE_T, s_aft))
            ax.scatter(t_ann, sc, color=_COX_CLR, s=16, marker="o", zorder=6)
            ax.scatter(t_ann, sa, color=_AFT_CLR, s=16, marker="o", zorder=6)
            if t_ann in (12., 24., 36., 60.):
                ax.text(t_ann + 0.8, min(sc + 0.035, 1.02), f"{sc*100:.0f}%",
                        ha="left", va="center", fontsize=6.2, color=_COX_CLR)
                ax.text(t_ann + 0.8, max(sa - 0.035, 0.04), f"{sa*100:.0f}%",
                        ha="left", va="center", fontsize=6.2, color=_AFT_CLR)

        for label, s_arr, clr, y_txt in [
            ("Cox median", s_cox, _COX_CLR, 0.12),
            ("AFT median", s_aft, _AFT_CLR, 0.06),
        ]:
            med_t = _median_crossing(_CURVE_T, s_arr)
            if med_t is not None and med_t <= 72:
                ax.vlines(med_t, 0.0, 0.5, color=clr, lw=0.8,
                          linestyles=":", zorder=2)
                ax.text(med_t + 1.0, y_txt, f"{label}: {med_t:.0f}m",
                        ha="left", va="center", fontsize=6.7, color=clr)

        ax.set_xlim(0, 72)
        ax.set_ylim(0, 1.06)
        ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Survival probability")
        ax.legend(
            fontsize=8.0,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.82)

        return fig

    # ── Probability table ─────────────────────────────────────────────────────

    @render.ui
    def prob_table():
        s_cox, s_aft = curves()
        rows = ""
        for t in _KEY_T:
            sc = float(np.interp(t, _CURVE_T, s_cox))
            sa = float(np.interp(t, _CURVE_T, s_aft))
            diff = sa - sc
            diff_str = (f'+{diff*100:.1f}%' if diff >= 0 else f'{diff*100:.1f}%')
            rows += (
                f'<tr>'
                f'<td>{int(t):d} m</td>'
                f'<td class="val cox">{sc*100:.1f}%</td>'
                f'<td class="val aft">{sa*100:.1f}%</td>'
                f'<td style="color:{BRAND["green"] if diff>=0 else BRAND["red"]};'
                f'font-size:.67rem;font-weight:600;">{diff_str}</td>'
                f'</tr>'
            )
        return ui.HTML(f"""
<div style="overflow-x:auto;margin-top:4px;">
  <table class="prob-tbl">
    <thead><tr>
      <th>Time</th>
      <th style="color:{_COX_CLR};">Cox PH</th>
      <th style="color:{_AFT_CLR};">AFT</th>
      <th>AFT - Cox</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
<p style="font-size:.64rem;color:{_MUTED};margin-top:8px;line-height:1.42;">
  Right-side values show AFT minus Cox PH at each horizon.<br>
  Both models are consistent with population-level TCGA-LUAD survival data.
</p>
""")

    # ── Distribution fitting plot (static) ────────────────────────────────────

    @render.plot
    def dist_plot():
        fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=300)
        _cell_ax(fig, ax)

        km_t, km_s = _survival_function_frame(KM_TRAIN)
        t_max = min(240.0, float(np.nanmax(km_t)) if km_t is not None else 240.0)
        t_grid = np.linspace(0.0, t_max, 260)

        handles = []
        labels = []
        if km_t is not None:
            km_line = ax.step(km_t, km_s, where="post", color=BRAND["ink"],
                              lw=1.0, label="Kaplan-Meier")[0]
            ci = getattr(KM_TRAIN, "confidence_interval_", None)
            if ci is not None and not ci.empty and ci.shape[1] >= 2:
                ci_t = ci.index.to_numpy(dtype=float)
                lo = ci.iloc[:, 0].to_numpy(dtype=float)
                hi = ci.iloc[:, 1].to_numpy(dtype=float)
                ax.fill_between(ci_t, lo, hi, step="post", color=BRAND["ink"],
                                alpha=0.08, linewidth=0, zorder=0)
            handles.append(km_line)
            labels.append("Kaplan-Meier")

        colors = [BRAND["blue"], _AFT_CLR, BRAND["green"], BRAND["red"]]
        for (name, fitter), clr in zip(DIST["fitters"].items(), colors):
            is_best = (name == DIST["best"])
            sf_vals = _survival_at_times(fitter, t_grid)
            line = ax.plot(
                t_grid, sf_vals, color=clr, lw=1.0,
                ls="-" if is_best else "--",
                alpha=1.0 if is_best else 0.68,
                label=f"{name}{' (best)' if is_best else ''}",
                zorder=3 if is_best else 2,
            )[0]
            handles.append(line)
            labels.append(f"{name}{' (best)' if is_best else ''}")

        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Survival Probability")
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, 1.05)
        ax.legend(handles, labels, loc="upper right", frameon=False,
                  fontsize=8.0, handlelength=2.0, borderaxespad=0.4)
        fig.tight_layout(pad=0.7)
        return fig

    # ── Distribution AIC table ────────────────────────────────────────────────

    @render.ui
    def dist_table():
        tbl = DIST["table"]
        rows = ""
        for _, row in tbl.iterrows():
            is_best = row["Distribution"] == DIST["best"]
            cls = " class='best'" if is_best else ""
            star = " ★" if is_best else ""
            rows += (
                f"<tr{cls}>"
                f"<td><strong>{row['Distribution']}{star}</strong></td>"
                f"<td class='num'>{row['AIC']:.1f}</td>"
                f"<td class='num'>{row['BIC']:.1f}</td>"
                f"<td class='num'>{row['Median (months)']:.1f}</td>"
                f"</tr>"
            )
        return ui.HTML(f"""
<div style="overflow-x:auto;margin-top:4px;">
  <table class="mtbl">
    <thead><tr>
      <th>Distribution</th><th>AIC</th><th>BIC</th><th>Median OS (m)</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
""")

    # ── Model performance plot (static, 2-panel) ──────────────────────────────

    def _make_perf_plot(is_narrow: bool):
        if is_narrow:
            fig = plt.figure(figsize=(3.5, 5.0), dpi=300)
            gs = fig.add_gridspec(2, 1, hspace=0.70)
            axs = [fig.add_subplot(gs[i, 0]) for i in range(2)]
        else:
            fig = plt.figure(figsize=(7.0, 3.5), dpi=300)
            gs = fig.add_gridspec(1, 2, wspace=0.42)
            axs = [fig.add_subplot(gs[0, i]) for i in range(2)]

        for ax in axs:
            _cell_ax(fig, ax)

        # ── A: C-index forest plot ────────────────────────────────────────────
        ax = axs[0]
        names  = ["Cox PH", f"{DIST['best']} AFT"]
        ci_v   = [RES_COX["c_index"],  RES_AFT["c_index"]]
        ci_lo  = [RES_COX["ci_lo"],    RES_AFT["ci_lo"]]
        ci_hi  = [RES_COX["ci_hi"],    RES_AFT["ci_hi"]]
        colors = [_COX_CLR, _AFT_CLR]

        ax.axvline(0.5, color=BRAND["edge"], lw=0.8, ls="--", zorder=0)
        for i, (n, c, lo, hi, clr) in enumerate(
                zip(names, ci_v, ci_lo, ci_hi, colors)):
            xerr = np.array([[c - lo], [hi - c]])
            ax.errorbar(c, i, xerr=xerr, fmt="o", color=clr, ecolor=clr,
                        elinewidth=0.8, capsize=3, markersize=4, zorder=5)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.0)
        x_min = max(0.45, min(ci_lo) - 0.05)
        x_max = min(0.92, max(ci_hi) + 0.08)
        ax.set_xlim(x_min, x_max)
        for i, (c, hi, clr) in enumerate(zip(ci_v, ci_hi, colors)):
            x_pos = min(hi + 0.010, x_max - 0.012)
            ha = "left" if x_pos < x_max - 0.02 else "right"
            ax.text(x_pos, i, f"{c:.3f}", va="center", ha=ha, fontsize=7.6,
                    color=clr, fontweight="600")
        ax.set_xlabel("C-index (95% CI)", labelpad=4)
        ax.set_title("A. C-index", fontweight="bold", loc="left",
                     fontsize=10.0, pad=6)
        ax.tick_params(axis="y", pad=3)

        # ── B: Time-dependent AUC ─────────────────────────────────────────────
        ax = axs[1]
        auc_min = 1.0
        auc_max = 0.5
        for res, clr, nm in [
            (RES_COX, _COX_CLR, "Cox"),
            (RES_AFT, _AFT_CLR, "AFT"),
        ]:
            if not any(np.isnan(res["auc_vals"])):
                auc_vals = np.array(res["auc_vals"], dtype=float)
                auc_min = min(auc_min, float(np.nanmin(auc_vals)))
                auc_max = max(auc_max, float(np.nanmax(auc_vals)))
                ax.plot(res["times"], auc_vals, color=clr, lw=1.1,
                        marker="o", ms=4,
                        label=f"{nm} mean={res['mean_auc']:.3f}")

        ax.axhline(0.5, color=_MUTED, lw=0.8, ls="--")
        ax.set_ylim(max(0.45, auc_min - 0.08), min(1.0, auc_max + 0.10))
        ax.set_xlabel("Time (months)", labelpad=5)
        ax.set_ylabel("Dynamic AUC", labelpad=2)
        ax.set_title("B. Time-dependent AUC", fontweight="bold",
                     loc="left", fontsize=10.0, pad=6)
        ax.legend(fontsize=8.0, frameon=False, loc="lower right",
                  handlelength=2.2, borderaxespad=0.2)
        if is_narrow:
            fig.subplots_adjust(
                left=0.22, right=0.95, top=0.97, bottom=0.09, hspace=0.70,
            )
        else:
            fig.subplots_adjust(
                left=0.095, right=0.985, top=0.90, bottom=0.18, wspace=0.42,
            )

        return fig

    def _figure_tag(fig, alt: str):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return ui.tags.img(
            src=f"data:image/png;base64,{encoded}",
            alt=alt,
        )

    @render.ui
    def perf_plot_desktop():
        return _figure_tag(
            _make_perf_plot(False),
            "C-index and time-dependent AUC model performance comparison",
        )

    @render.ui
    def perf_plot_mobile():
        return _figure_tag(
            _make_perf_plot(True),
            "C-index and time-dependent AUC model performance comparison",
        )

    # ── Performance metric chips ──────────────────────────────────────────────

    @render.ui
    def perf_chips():
        def _fmt(v, decimals=3):
            return f"{v:.{decimals}f}" if isinstance(v, float) and not np.isnan(v) else "—"

        def _chip(label, train_v, test_v, clr, decimals=3):
            return (
                f'<div class="metric-chip">'
                f'<span class="mc-label">{label}</span>'
                f'<span class="mc-value" style="color:{clr};">'
                f'{_fmt(train_v, decimals)} / {_fmt(test_v, decimals)}</span>'
                f'<span style="font-size:.60rem;color:{_MUTED};">train / test</span>'
                f'</div>'
            )

        chips = "".join([
            _chip("Cox PH — C-index",
                  TR_COX.get("c_index", float("nan")), RES_COX["c_index"], _COX_CLR),
            _chip("Cox PH — IBS",
                  TR_COX.get("ibs", float("nan")), RES_COX["ibs"], _COX_CLR, 4),
            f'<div class="metric-chip">'
            f'<span class="mc-label">Cox PH — mean AUC</span>'
            f'<span class="mc-value mc-cox">{_fmt(RES_COX["mean_auc"])}</span>'
            f'<span style="font-size:.60rem;color:{_MUTED};">test only</span>'
            f'</div>',
            _chip(f"AFT — C-index",
                  TR_AFT.get("c_index", float("nan")), RES_AFT["c_index"], _AFT_CLR),
            _chip(f"AFT — IBS",
                  TR_AFT.get("ibs", float("nan")), RES_AFT["ibs"], _AFT_CLR, 4),
            f'<div class="metric-chip">'
            f'<span class="mc-label">AFT — mean AUC</span>'
            f'<span class="mc-value mc-aft">{_fmt(RES_AFT["mean_auc"])}</span>'
            f'<span style="font-size:.60rem;color:{_MUTED};">test only</span>'
            f'</div>',
        ])
        return ui.HTML(
            f'<div class="infobar" style="margin-top:10px;">{chips}</div>'
        )

    # ── Methods panel ─────────────────────────────────────────────────────────

    @render.ui
    def methods_panel():
        feat_rows = "".join(
            f"<tr><td><code>{c}</code></td>"
            f"<td>{FEAT_DISPLAY[c]}</td>"
            f"<td class='num'>{float(_IMP.statistics_[i]):.1f}</td></tr>"
            for i, c in enumerate(FEAT_COLS)
        )

        def _f(v, d=3):
            return f"{v:.{d}f}" if isinstance(v, float) and not np.isnan(v) else "—"

        cmp_rows = ""
        best_c = max(RES_COX["c_index"], RES_AFT["c_index"])
        for nm, res, tr, clr in [
            ("Cox PH", RES_COX, TR_COX, _COX_CLR),
            (f"{DIST['best']} AFT", RES_AFT, TR_AFT, _AFT_CLR),
        ]:
            is_best = abs(res["c_index"] - best_c) < 1e-9
            bold = f"font-weight:700;color:{clr};" if is_best else ""
            star = " ★" if is_best else ""
            cmp_rows += (
                f'<tr style="{bold}">'
                f"<td>{nm}{star}</td>"
                f"<td class='num'>{_f(tr.get('c_index', float('nan')))} / {_f(res['c_index'])}</td>"
                f"<td class='num'>[{_f(res['ci_lo'])}, {_f(res['ci_hi'])}]</td>"
                f"<td class='num'>{_f(tr.get('ibs', float('nan')), 4)} / {_f(res['ibs'], 4)}</td>"
                f"<td class='num'>{_f(res['mean_auc'])}</td>"
                f"</tr>"
            )

        return ui.HTML(f"""
<div class="methods">

  <div class="card" style="margin-bottom:14px;">
    <div class="card-header">Model Comparison — Train / Test
      &nbsp;<span style="font-weight:400;font-size:.78em;color:{_MUTED};">
        Train N={N_TRAIN} · Test N={N_TEST} · ★ = best test C-index</span>
    </div>
    <div class="card-body" style="padding:14px!important;">
      <table class="mtbl">
        <thead><tr>
          <th>Model</th><th>C-index (train / test)</th><th>95% Bootstrap CI</th>
          <th>IBS (train / test)</th><th>Mean AUC (test)</th>
        </tr></thead>
        <tbody>{cmp_rows}</tbody>
      </table>
      <p style="font-size:.74rem;color:{_MUTED};margin-top:8px;">
        Training metrics are apparent performance (no resampling).
        A large train–test gap indicates overfitting.
      </p>
    </div>
  </div>

  <div class="card" style="margin-bottom:14px;">
    <div class="card-header">Methods</div>
    <div class="card-body" style="padding:14px!important;">
      <h4>Data</h4>
      <p>TCGA-LUAD (The Cancer Genome Atlas — Lung Adenocarcinoma).
      {N_TOTAL} patients downloaded from the NCI Genomic Data Commons (GDC) public API,
      no registration required.
      Outcome: overall survival (OS) — days to death or last follow-up converted to months.
      Event rate {EV_RATE:.0%} · median follow-up {MED_FU:.0f} months.
      Split 80/20 (stratified by event) → train N={N_TRAIN}, test N={N_TEST}.
      Missing values imputed by training-set median.</p>

      <h4>Distribution selection</h4>
      <p>Four parametric families (Weibull, Log-Normal, Log-Logistic, Exponential) were
      fitted to the marginal survival times using maximum likelihood with censoring.
      AIC was used to select the AFT family. Best fit: <strong>{DIST['best']}</strong>
      (AIC advantage over Weibull:
      {DIST['table'].set_index('Distribution').loc['Weibull','AIC'] -
       DIST['table'].set_index('Distribution').loc[DIST['best'],'AIC']:.1f} points).</p>

      <h4>Cox Proportional Hazards</h4>
      <p>Semi-parametric model (lifelines). L2 regularisation — penaliser tuned by
      5-fold CV grid search over [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0].
      Selected penaliser: {COX._penalizer_used}. Survival curves computed via
      Breslow baseline hazard estimator.</p>

      <h4>{DIST['best']} Accelerated Failure Time</h4>
      <p>Fully parametric AFT model (lifelines). The AFT framework models survival time
      directly: log(T) = Xβ + σε, where ε follows the {DIST['best']} error distribution.
      Covariate effects are multiplicative on survival time — a hazard ratio analogue is
      not required. Closed-form S(t|x) allows reliable extrapolation beyond the observed
      follow-up period.</p>

      <h4>Evaluation</h4>
      <p>C-index (Harrell's concordance) with 95% bootstrap CI (150 resamples).
      Integrated Brier Score (IBS) over [12, 24, 36, 48, 60] months.
      Time-dependent AUC (cumulative / dynamic definition).</p>
    </div>
  </div>

  <div class="card">
    <div class="card-header">Features ({len(FEAT_COLS)} clinical variables)</div>
    <div class="card-body" style="padding:14px!important;">
      <table class="mtbl">
        <thead><tr>
          <th>Column</th><th>Description</th><th>Training Median</th>
        </tr></thead>
        <tbody>{feat_rows}</tbody>
      </table>
      <p style="font-size:.76rem;color:{_MUTED};margin-top:10px;">
        Missing values: imputed by training-set median.
        Stage distribution: I={int(STAGE_COUNTS.get("1", 0))}
        · II={int(STAGE_COUNTS.get("2", 0))}
        · III={int(STAGE_COUNTS.get("3", 0))}
        · IV={int(STAGE_COUNTS.get("4", 0))}.
      </p>
    </div>
  </div>

</div>
""")


app = App(app_ui, server)
