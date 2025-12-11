# ===== File: app.py =====
"""
Microplastics Risk Model API
- Adds /predict_csv to score CSVs and return a new CSV.
- Keeps EDA/modeling/persistence endpoints from previous version.

Run (local):
  uvicorn app:app --reload

Run (Docker):
  docker build -t mp-risk:latest .
  docker run --rm -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts mp-risk:latest

CSV schema (required):
  concentration_p_per_L, mean_size_um, fraction_lt100um, polymer, shape, route,
  ingestion_rate_L_per_day, body_weight_kg, exposure_days (optional)
"""

from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from math import exp
from io import StringIO, BytesIO
import csv
import os
import argparse
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Query, Path
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, validator, root_validator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ML stack
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

try:
    import joblib
except Exception:
    joblib = None

APP_VERSION = "0.5.0"

# ----------------- Risk model configuration -----------------
class ModelConfig(BaseModel):
    polymer_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "PVC": 1.00, "PS": 0.85, "PC": 0.80, "Nylon": 0.70,
            "PET": 0.60, "PP": 0.50, "PE": 0.45, "Other": 0.55,
        }
    )
    shape_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "fiber": 1.00, "fragment": 0.75, "sphere": 0.60,
            "film": 0.70, "foam": 0.65, "other": 0.70,
        }
    )
    route_multipliers: Dict[str, float] = Field(
        default_factory=lambda: {"ingestion": 1.00, "inhalation": 1.10, "dermal": 0.20}
    )
    logistic_k: float = 2.2
    logistic_x0: float = 1.0
    low_threshold: float = 0.33
    high_threshold: float = 0.66
    size50_ref_um: float = 100.0
    size_min_cap_um: float = 1.0
    max_exposure_index: float = 50.0
    class Config: extra = "forbid"

CONFIG = ModelConfig()

# ----------------- Schemas -----------------
class Route(str, Enum):
    ingestion = "ingestion"
    inhalation = "inhalation"
    dermal = "dermal"

class Shape(str, Enum):
    fiber = "fiber"
    fragment = "fragment"
    sphere = "sphere"
    film = "film"
    foam = "foam"
    other = "other"

class RiskRequest(BaseModel):
    concentration_p_per_L: float = Field(..., gt=0)
    mean_size_um: float = Field(..., gt=0)
    fraction_lt100um: float = Field(..., ge=0, le=1)
    polymer: str
    shape: Shape
    route: Route
    ingestion_rate_L_per_day: float = Field(..., gt=0)
    body_weight_kg: float = Field(..., gt=0)
    exposure_days: Optional[int] = Field(365, gt=0)
    @validator("polymer")
    def norm_polymer(cls, v: str) -> str: return v.strip()

class RiskComponent(BaseModel):
    exposure_index: float
    hazard_index: float
    bioavailability_index: float

class RiskResponse(BaseModel):
    score: float = Field(..., ge=0, le=1)
    category: str
    components: RiskComponent
    inputs: RiskRequest
    notes: List[str]

class BatchRiskRequest(BaseModel):
    items: Optional[List[RiskRequest]] = None

class ConfigPatch(BaseModel):
    polymer_weights: Optional[Dict[str, float]] = None
    shape_weights: Optional[Dict[str, float]] = None
    route_multipliers: Optional[Dict[str, float]] = None
    logistic_k: Optional[float] = Field(None, gt=0)
    logistic_x0: Optional[float] = None
    low_threshold: Optional[float] = Field(None, ge=0, le=1)
    high_threshold: Optional[float] = Field(None, ge=0, le=1)
    size50_ref_um: Optional[float] = Field(None, gt=0)
    size_min_cap_um: Optional[float] = Field(None, gt=0)
    max_exposure_index: Optional[float] = Field(None, gt=0)
    @root_validator
    def validate_thresholds(cls, values):
        lo, hi = values.get("low_threshold"), values.get("high_threshold")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("low_threshold must be < high_threshold")
        return values

# ----------------- Risk math -----------------
def _polymer_weight(polymer: str) -> Tuple[float, List[str]]:
    pkey = polymer.strip()
    if pkey in CONFIG.polymer_weights: return CONFIG.polymer_weights[pkey], []
    tkey = pkey.title()
    if tkey in CONFIG.polymer_weights: return CONFIG.polymer_weights[tkey], []
    return CONFIG.polymer_weights["Other"], [f"Polymer '{polymer}' not in table; used 'Other'."]

def _size_factor(mean_size_um: float, fraction_lt100um: float) -> float:
    size = max(mean_size_um, CONFIG.size_min_cap_um)
    base = CONFIG.size50_ref_um / size
    return min(base * (1 + 0.8 * fraction_lt100um), 10.0)

def exposure_index(req: RiskRequest) -> float:
    dose = (req.concentration_p_per_L * req.ingestion_rate_L_per_day) / req.body_weight_kg
    route_mult = CONFIG.route_multipliers.get(req.route.value, 1.0)
    ex = dose * route_mult
    return min(ex, CONFIG.max_exposure_index)

def hazard_index(req: RiskRequest, notes: List[str]) -> float:
    p_w, p_notes = _polymer_weight(req.polymer)
    if p_notes: notes.extend(p_notes)
    s_w = CONFIG.shape_weights.get(req.shape.value, CONFIG.shape_weights["other"])
    return (p_w * 0.65) + (s_w * 0.35)

def bioavailability_index(req: RiskRequest) -> float:
    return _size_factor(req.mean_size_um, req.fraction_lt100um)

def _logistic(x: float, k: float, x0: float) -> float:
    return 1.0 / (1.0 + exp(-k * (x - x0)))

def risk_score(exposure: float, hazard: float, bioavail: float) -> float:
    raw = exposure * hazard * bioavail
    x = raw ** 0.5
    return _logistic(x, CONFIG.logistic_k, CONFIG.logistic_x0)

def risk_category(score: float) -> str:
    if score < CONFIG.low_threshold: return "low"
    if score < CONFIG.high_threshold: return "moderate"
    return "high"

# ----------------- App -----------------
app = FastAPI(
    title="Microplastics Risk Model API",
    version=APP_VERSION,
    description="Risk scoring + EDA + Modeling + Persistence + CSV scoring",
)

@app.get("/", response_class=PlainTextResponse)
def root():
    return "Microplastics Risk Model API. See /docs."

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}

@app.get("/config")
def get_config(): return CONFIG.dict()

@app.post("/config")
def update_config(patch: ConfigPatch):
    data = patch.dict(exclude_unset=True)
    if "polymer_weights" in data: _validate_weights(data["polymer_weights"], "polymer_weights"); CONFIG.polymer_weights.update(data["polymer_weights"])
    if "shape_weights" in data: _validate_weights(data["shape_weights"], "shape_weights"); CONFIG.shape_weights.update(data["shape_weights"])
    if "route_multipliers" in data: _validate_weights(data["route_multipliers"], "route_multipliers"); CONFIG.route_multipliers.update(data["route_multipliers"])
    for k in ["logistic_k","logistic_x0","low_threshold","high_threshold","size50_ref_um","size_min_cap_um","max_exposure_index"]:
        if k in data: setattr(CONFIG, k, data[k])
    if CONFIG.low_threshold >= CONFIG.high_threshold:
        raise HTTPException(status_code=400, detail="low_threshold must be < high_threshold")
    return CONFIG.dict()

def _validate_weights(d: Dict[str, float], label: str):
    for k, v in d.items():
        if not isinstance(v, (int, float)):
            raise HTTPException(status_code=400, detail=f"{label}.{k} must be numeric")
        if v <= 0:
            raise HTTPException(status_code=400, detail=f"{label}.{k} must be > 0")

@app.post("/risk", response_model=RiskResponse)
def compute_risk(req: RiskRequest):
    notes: List[str] = []
    ex = exposure_index(req)
    hz = hazard_index(req, notes)
    ba = bioavailability_index(req)
    score = risk_score(exposure=ex, hazard=hz, bioavail=ba)
    cat = risk_category(score)
    notes.extend(_advisories(req, ex, hz, ba))
    return RiskResponse(
        score=round(score, 4),
        category=cat,
        components=RiskComponent(
            exposure_index=round(ex, 4),
            hazard_index=round(hz, 4),
            bioavailability_index=round(ba, 4),
        ),
        inputs=req,
        notes=notes,
    )

@app.post("/batch")
async def batch_risk(
    json_body: Optional[BatchRiskRequest] = Body(None),
    file: Optional[UploadFile] = File(None),
):
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    if file is not None:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv accepted.")
        content = (await file.read()).decode("utf-8", errors="replace")
        for i, row in enumerate(_csv_iter(content), start=1):
            try:
                req = RiskRequest(**row)
                resp = compute_risk(req)
                results.append(resp.dict())
            except Exception as e:
                errors.append({"row": i, "error": str(e), "data": row})
    elif json_body and json_body.items:
        for i, item in enumerate(json_body.items, start=1):
            try:
                resp = compute_risk(item)
                results.append(resp.dict())
            except Exception as e:
                errors.append({"index": i, "error": str(e), "data": item.dict()})
    else:
        raise HTTPException(status_code=400, detail="Provide JSON 'items' or CSV file.")
    return {"count": len(results), "results": results, "errors": errors}

# ---------- NEW: /predict_csv (score CSV -> return CSV) ----------
@app.post("/predict_csv")
async def predict_csv(
    file: UploadFile = File(...),
    include_components: bool = Query(False, description="Include exposure/hazard/bioavailability columns"),
    round_score: int = Query(4, ge=0, le=10),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv is accepted.")
    text = (await file.read()).decode("utf-8", errors="replace")
    reader = _csv_iter(text)

    out_io = StringIO()
    writer = None

    for row in reader:
        req = RiskRequest(**row)
        ex = exposure_index(req)
        hz = hazard_index(req, [])
        ba = bioavailability_index(req)
        score = risk_score(ex, hz, ba)
        cat = risk_category(score)

        # Initialize CSV header on first row
        if writer is None:
            base_headers = list(row.keys())
            extra = ["risk_score", "risk_level"]
            comps = ["exposure_index", "hazard_index", "bioavailability_index"] if include_components else []
            writer = csv.DictWriter(out_io, fieldnames=base_headers + extra + comps)
            writer.writeheader()

        out_row = dict(row)
        out_row["risk_score"] = round(score, round_score)
        out_row["risk_level"] = cat
        if include_components:
            out_row["exposure_index"] = round(ex, round_score)
            out_row["hazard_index"] = round(hz, round_score)
            out_row["bioavailability_index"] = round(ba, round_score)
        writer.writerow(out_row)

    # stream back
    mem = BytesIO(out_io.getvalue().encode("utf-8"))
    headers = {"Content-Disposition": f'attachment; filename="scored_{file.filename}"'}
    return StreamingResponse(mem, media_type="text/csv", headers=headers)

# ----------------- Preprocessing & EDA -----------------
NUMERIC_COLS = [
    "concentration_p_per_L", "mean_size_um", "fraction_lt100um",
    "ingestion_rate_L_per_day", "body_weight_kg"
]
CATEGORICAL_COLS = ["polymer", "shape", "route"]
SKEW_WHITELIST = ["concentration_p_per_L", "mean_size_um"]

@dataclass
class PreprocessReport:
    scaling: str
    encoding: str
    outliers: str
    iqr_k: float
    winsorized: Dict[str, Dict[str, float]]
    removed_rows: int
    skewed: Dict[str, float]
    transformed: List[str]

def encode_categoricals_pd(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, dummy_na=False, drop_first=False)

def scale_features_pd(df: pd.DataFrame, cols: List[str], method: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    stats: Dict[str, Dict[str, float]] = {}
    df_scaled = df.copy()
    if method == "none": return df_scaled, stats
    for c in cols:
        x = df_scaled[c].astype(float)
        if method == "standard":
            mu, sd = float(np.nanmean(x)), float(np.nanstd(x, ddof=0) or 1.0)
            df_scaled[c] = (x - mu) / (sd or 1.0)
            stats[c] = {"mean": mu, "std": sd}
        elif method == "minmax":
            mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
            rng = (mx - mn) if (mx - mn) != 0 else 1.0
            df_scaled[c] = (x - mn) / rng
            stats[c] = {"min": mn, "max": mx}
        else:
            raise ValueError("scaling must be 'standard' | 'minmax' | 'none'")
    return df_scaled, stats

def handle_outliers(df: pd.DataFrame, cols: List[str], mode: str, iqr_k: float) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], int]:
    wins: Dict[str, Dict[str, float]] = {}
    if mode == "none": return df, wins, 0
    work = df.copy()
    bounds = {}
    for c in cols:
        x = work[c].astype(float)
        q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
        iqr = q3 - q1
        lb, ub = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        bounds[c] = (lb, ub)
    if mode == "winsorize":
        for c, (lb, ub) in bounds.items():
            work[c] = np.clip(work[c].astype(float), lb, ub)
            wins[c] = {"lower": float(lb), "upper": float(ub)}
        return work, wins, 0
    if mode == "remove":
        mask = np.ones(len(work), dtype=bool)
        for c, (lb, ub) in bounds.items():
            x = work[c].astype(float)
            mask &= (x >= lb) & (x <= ub)
        removed = int((~mask).sum())
        return work.loc[mask].reset_index(drop=True), wins, removed
    raise ValueError("outliers must be 'winsorize' | 'remove' | 'none'")

def transform_skew(df: pd.DataFrame, cols: List[str], mode: str) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    if mode == "none": return df, {}, []
    skewness = df[cols].apply(lambda s: float(s.astype(float).skew())).to_dict()
    to_tx = [c for c in cols if (abs(skewness.get(c, 0.0)) > 1.0)]
    work = df.copy()
    transformed = []
    for c in to_tx:
        if (work[c] <= -1).any():  # safety for log1p
            continue
        work[c] = np.log1p(work[c].astype(float))
        transformed.append(c)
    return work, skewness, transformed

def add_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        req = RiskRequest(
            concentration_p_per_L=float(r["concentration_p_per_L"]),
            mean_size_um=float(r["mean_size_um"]),
            fraction_lt100um=float(r["fraction_lt100um"]),
            polymer=str(r["polymer"]),
            shape=Shape(str(r["shape"]).lower()),
            route=Route(str(r["route"]).lower()),
            ingestion_rate_L_per_day=float(r["ingestion_rate_L_per_day"]),
            body_weight_kg=float(r["body_weight_kg"]),
            exposure_days=int(r.get("exposure_days", 365)),
        )
        ex = exposure_index(req)
        hz = hazard_index(req, [])
        ba = bioavailability_index(req)
        score = risk_score(ex, hz, ba)
        rows.append({"risk_score": score, "risk_level": risk_category(score)})
    out = df.copy()
    out["risk_score"] = [r["risk_score"] for r in rows]
    out["risk_level"] = [r["risk_level"] for r in rows]
    return out

def summarize_by_level(df: pd.DataFrame) -> Dict[str, Any]:
    g = df.groupby("risk_level")["risk_score"]
    desc = g.describe().to_dict()
    samples = [grp["risk_score"].values for _, grp in df.groupby("risk_level")]
    kw_stat, kw_p = _kruskal_wallis(samples)
    return {"describe": desc, "kruskal_wallis": {"H": kw_stat, "p_value": kw_p}}

def _kruskal_wallis(samples: List[np.ndarray]) -> Tuple[float, Optional[float]]:
    if len(samples) < 2: return 0.0, None
    all_vals = np.concatenate(samples)
    ranks = _rankdata(all_vals)
    split = np.cumsum([len(s) for s in samples])[:-1]
    ranks_groups = np.split(ranks, split)
    n_total = len(all_vals)
    tie_correction = _tie_correction(all_vals)
    ss = 0.0
    for r, s in zip(ranks_groups, samples):
        ss += (r.sum()**2) / len(s)
    H = (12 / (n_total * (n_total + 1)))*ss - 3*(n_total + 1)
    H = H / tie_correction if tie_correction != 0 else H
    return float(H), None

def _rankdata(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a)+1)
    _, idx, counts = np.unique(a[order], return_inverse=True, return_counts=True)
    cumsum = np.cumsum(counts)
    starts = cumsum - counts + 1
    avg = (starts + cumsum) / 2.0
    ranks[order] = avg[idx]
    return ranks

def _tie_correction(a: np.ndarray) -> float:
    _, counts = np.unique(a, return_counts=True)
    return 1.0 - (np.sum(counts**3 - counts)) / (len(a)**3 - len(a))

def correlate_risk_vs_conc(df: pd.DataFrame) -> Dict[str, float]:
    x = df["concentration_p_per_L"].astype(float).values
    y = df["risk_score"].astype(float).values
    px = (x - x.mean()) / (x.std() or 1.0)
    py = (y - y.mean()) / (y.std() or 1.0)
    pearson = float(np.clip(np.corrcoef(px, py)[0,1], -1, 1))
    rx = _rankdata(x); ry = _rankdata(y)
    spearman = float(np.clip(np.corrcoef(rx, ry)[0,1], -1, 1))
    X = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    slope = float(beta[1]); intercept = float(beta[0])
    return {"pearson": pearson, "spearman": spearman, "slope": slope, "intercept": intercept}

def make_artifacts_dir(path: str = "./artifacts") -> str:
    os.makedirs(path, exist_ok=True); return path

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def list_artifacts() -> List[str]:
    outdir = make_artifacts_dir()
    return sorted([f for f in os.listdir(outdir) if os.path.isfile(os.path.join(outdir, f))])

def plot_risk_distribution(df: pd.DataFrame, outdir: str) -> str:
    fig, ax = plt.subplots()
    ax.hist(df["risk_score"].astype(float).values, bins=30)
    ax.set_title("Distribution of Risk Score")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Count")
    out = os.path.join(outdir, "plot_risk_distribution.png")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

def plot_risk_by_level(df: pd.DataFrame, outdir: str) -> str:
    fig, ax = plt.subplots()
    order = ["low", "moderate", "high"]
    data = [df.loc[df["risk_level"]==lvl, "risk_score"].values for lvl in order]
    ax.boxplot(data, labels=order)
    ax.set_title("Risk Score by Risk Level")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Risk Score")
    out = os.path.join(outdir, "plot_risk_by_level.png")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

def plot_risk_vs_conc(df: pd.DataFrame, outdir: str) -> str:
    fig, ax = plt.subplots()
    x = df["concentration_p_per_L"].astype(float).values
    y = df["risk_score"].astype(float).values
    ax.scatter(x, y)
    X = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    xs = np.linspace(x.min(), x.max(), 100)
    ys = beta[0] + beta[1]*xs
    ax.plot(xs, ys)
    ax.set_title("Risk Score vs MP Count per L")
    ax.set_xlabel("MP count per L")
    ax.set_ylabel("Risk Score")
    out = os.path.join(outdir, "plot_risk_vs_concentration.png")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

def plot_polymer_distribution(df: pd.DataFrame, outdir: str) -> str:
    fig, ax = plt.subplots()
    counts = df["polymer"].value_counts()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Polymer Type Distribution")
    ax.set_xlabel("Polymer")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)
    out = os.path.join(outdir, "plot_polymer_distribution.png")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

def _advisories(req: RiskRequest, ex: float, hz: float, ba: float) -> List[str]:
    adv: List[str] = []
    if req.mean_size_um < 20: adv.append("High bioavailability expected for <20 µm particles.")
    if req.fraction_lt100um > 0.8: adv.append("Dominance of <100 µm fraction elevates uptake likelihood.")
    if ex > CONFIG.max_exposure_index * 0.9: adv.append("Exposure index reached clamp; verify inputs.")
    if req.polymer.strip() not in CONFIG.polymer_weights: adv.append("Consider adding polymer-specific weight via /config.")
    return adv

# ----------------- /analyze (EDA) -----------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    scaling: str = Query("standard"),
    outliers: str = Query("winsorize"),
    iqr_k: float = Query(1.5),
    skew: str = Query("auto"),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv is accepted.")
    content = (await file.read()).decode("utf-8", errors="replace")
    df = _csv_to_df(content)
    df = add_risk_columns(df)

    df, wins, removed = handle_outliers(df, NUMERIC_COLS, outliers, float(iqr_k))
    df, skewness, transformed = transform_skew(df, ["concentration_p_per_L","mean_size_um"], skew)
    df_e = encode_categoricals_pd(df, CATEGORICAL_COLS)
    df_s, stats = scale_features_pd(df_e, NUMERIC_COLS, scaling)

    by_level = summarize_by_level(df_s)
    corr = correlate_risk_vs_conc(df_s)

    outdir = make_artifacts_dir()
    processed_path = os.path.join(outdir, "processed.csv")
    df_s.to_csv(processed_path, index=False)

    p_dist = plot_risk_distribution(df_s, outdir)
    p_by = plot_risk_by_level(df_s, outdir)
    p_vs = plot_risk_vs_conc(df_s, outdir)
    p_poly = plot_polymer_distribution(df, outdir)

    report = PreprocessReport(
        scaling=scaling, encoding="onehot", outliers=outliers,
        iqr_k=float(iqr_k), winsorized=wins, removed_rows=removed,
        skewed=skewness, transformed=transformed
    )
    return {
        "preprocessing": report.__dict__,
        "scaling_stats": stats,
        "distribution": {"risk_score_hist": p_dist},
        "by_risk_level": by_level,
        "risk_vs_concentration": corr,
        "polymer_distribution": p_poly,
        "artifacts": {
            "processed_csv": processed_path,
            "plot_risk_distribution": p_dist,
            "plot_risk_by_level": p_by,
            "plot_risk_vs_concentration": p_vs,
            "plot_polymer_distribution": p_poly,
        },
        "rows_out": int(len(df_s)),
    }

# ----------------- Feature selection + Modeling -----------------
def build_preprocess_ct(scaling: str) -> ColumnTransformer:
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "none": "passthrough"}[scaling]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num", scaler, NUMERIC_COLS),
        ("cat", ohe, CATEGORICAL_COLS),
    ])

def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "risk_level" not in df.columns or "risk_score" not in df.columns:
        df = add_risk_columns(df)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df["risk_level"].astype(str)
    return X, y

_BEST_MODEL: Optional[Pipeline] = None
_BEST_MODEL_NAME: Optional[str] = None
_BEST_FEATURE_NAMES: Optional[List[str]] = None
_LAST_TRAIN_METRICS: Optional[Dict[str, Any]] = None

@app.post("/model/risk_type/train")
async def train_risk_type(
    file: UploadFile = File(...),
    scaling: str = Query("standard"),
    outliers: str = Query("winsorize"),
    iqr_k: float = Query(1.5),
    skew: str = Query("auto"),
    selector: str = Query("mi"),
    top_k: int = Query(20),
    smote: bool = Query(True),
    random_state: int = Query(42),
):
    global _BEST_MODEL, _BEST_MODEL_NAME, _BEST_FEATURE_NAMES, _LAST_TRAIN_METRICS

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv is accepted.")
    content = (await file.read()).decode("utf-8", errors="replace")
    raw = _csv_to_df(content)
    raw = add_risk_columns(raw)

    raw, _, _ = handle_outliers(raw, NUMERIC_COLS, outliers, float(iqr_k))
    raw, _, _ = transform_skew(raw, ["concentration_p_per_L","mean_size_um"], skew)
    X_df, y = prepare_xy(raw)

    ct = build_preprocess_ct(scaling)
    ct_fit = ct.fit(X_df)
    ohe = [t for n,t,c in ct_fit.transformers_ if n=="cat"][0]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    feat_names = NUMERIC_COLS + cat_names

    if smote and SMOTE is None:
        raise HTTPException(status_code=500, detail="imblearn not installed; set smote=false or install imblearn.")

    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=random_state, stratify=y)

    if smote:
        X_tr = ct_fit.transform(X_train_df)
        smoter = SMOTE(random_state=random_state)
        X_tr, y_train = smoter.fit_resample(X_tr, y_train)
        pre = "passthrough"
    else:
        X_tr = None
        pre = ct

    models = {
        "LogReg": LogisticRegression(max_iter=5000, multi_class="ovr", solver="lbfgs", random_state=random_state),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state),
        "GB": GradientBoostingClassifier(random_state=random_state),
    }
    logreg_grid = {"clf__C": [0.1, 0.5, 1.0, 2.0]}

    results = {}
    best_name, best_auc = None, -1.0
    best_pipe: Optional[Pipeline] = None

    for name, clf in models.items():
        steps = [("pre", pre)]
        if selector == "mi":
            steps += [("sel", SelectKBest(mutual_info_classif, k=min(top_k, len(feat_names))))]
        steps += [("clf", clf)]
        pipe = Pipeline(steps)

        if name == "LogReg":
            grid = GridSearchCV(pipe, param_grid=logreg_grid, cv=5, scoring="roc_auc_ovr")
            model = grid.fit(X_tr if smote else X_train_df, y_train)
            pipe = model.best_estimator_
        else:
            pipe = pipe.fit(X_tr if smote else X_train_df, y_train)

        X_test_processed = ct_fit.transform(X_test_df) if smote else X_test_df
        y_pred = pipe.predict(X_test_processed)
        y_proba = _proba_ovr(pipe, X_test_processed, classes=sorted(y.unique()))
        metrics = _eval_metrics(y_test, y_pred, y_proba, classes=sorted(y.unique()))
        results[name] = metrics

        if (metrics["roc_auc_ovr"] or -1.0) > best_auc:
            best_auc = (metrics["roc_auc_ovr"] or -1.0); best_name = name; best_pipe = pipe

    _BEST_MODEL = best_pipe
    _BEST_MODEL_NAME = best_name
    _BEST_FEATURE_NAMES = feat_names
    _LAST_TRAIN_METRICS = results

    outdir = make_artifacts_dir()
    perf_plot = _plot_model_performance(results, outdir)
    cm_plot = _plot_confusion_matrix(y_test, y_pred, outdir, f"cm_{best_name}.png")

    return {
        "selected_features": {"method": selector, "top_k": top_k},
        "class_imbalance": {"smote": smote},
        "results": results,
        "best_model": best_name,
        "artifacts": {"performance_bar": perf_plot, "confusion_matrix": cm_plot},
        "feature_names": feat_names[:50]
    }

def _proba_ovr(pipe: Pipeline, X, classes: List[str]) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)
        except Exception:
            pass
    n = len(X); k = len(classes)
    return np.ones((n, k)) / k

def _eval_metrics(y_true, y_pred, y_proba, classes: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
    except Exception:
        auc = float("nan")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    return {"accuracy": acc, "f1_macro": f1, "roc_auc_ovr": auc, "report": report, "confusion_matrix": {"labels": classes, "matrix": cm}}

def _plot_model_performance(results: Dict[str, Dict[str, Any]], outdir: str) -> str:
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["f1_macro"] for n in names]
    auc = [results[n]["roc_auc_ovr"] for n in names]

    fig, ax = plt.subplots()
    ax.bar(names, acc); ax.set_title("Accuracy by Model"); ax.set_ylabel("Accuracy"); ax.set_xlabel("Model")
    path_acc = os.path.join(outdir, "model_accuracy.png"); fig.tight_layout(); fig.savefig(path_acc, dpi=160); plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(names, f1); ax.set_title("F1-macro by Model"); ax.set_ylabel("F1-macro"); ax.set_xlabel("Model")
    path_f1 = os.path.join(outdir, "model_f1.png"); fig.tight_layout(); fig.savefig(path_f1, dpi=160); plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(names, auc); ax.set_title("ROC-AUC(ovr) by Model"); ax.set_ylabel("AUC"); ax.set_xlabel("Model")
    path_auc = os.path.join(outdir, "model_auc.png"); fig.tight_layout(); fig.savefig(path_auc, dpi=160); plt.close(fig)

    return path_acc

def _plot_confusion_matrix(y_true, y_pred, outdir: str, filename: str) -> str:
    labels = sorted(pd.Series(y_true).unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    out = os.path.join(outdir, filename)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

# ----------------- Importance (optional dataset) -----------------
@app.post("/model/risk_type/importance")
async def model_importance(file: Optional[UploadFile] = File(None), top_n: int = Query(15)):
    if _BEST_MODEL is None or _BEST_FEATURE_NAMES is None:
        raise HTTPException(status_code=400, detail="No trained model in memory. Train via /model/risk_type/train first.")
    if file is None:
        return {
            "best_model": _BEST_MODEL_NAME,
            "available_features_preview": _BEST_FEATURE_NAMES[:min(top_n, len(_BEST_FEATURE_NAMES))],
            "note": "Provide a CSV to compute permutation importances on your dataset."
        }

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv is accepted.")
    content = (await file.read()).decode("utf-8", errors="replace")
    df = _csv_to_df(content)
    df = add_risk_columns(df)
    X_df, y = prepare_xy(df)

    X = ColumnTransformer([
        ("num", "passthrough", NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), CATEGORICAL_COLS)
    ]).fit_transform(X_df)

    try:
        perm = permutation_importance(_BEST_MODEL, X, y, n_repeats=10, random_state=42, scoring="f1_macro")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Permutation importance failed: {e}")

    idx = np.argsort(perm.importances_mean)[::-1][:top_n]
    feats = np.array(_BEST_FEATURE_NAMES)[idx]
    means = perm.importances_mean[idx]
    stds = perm.importances_std[idx]

    outdir = make_artifacts_dir()
    fig, ax = plt.subplots()
    ax.barh(range(len(feats)), means[::-1])
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats[::-1])
    ax.set_title("Permutation Importance (top-N)")
    ax.set_xlabel("Importance (mean)")
    fig.tight_layout()
    imp_path = os.path.join(outdir, "permutation_importance.png")
    fig.savefig(imp_path, dpi=160); plt.close(fig)

    return {
        "best_model": _BEST_MODEL_NAME,
        "top_features": [{"feature": f, "mean": float(m), "std": float(s)} for f,m,s in zip(feats, means, stds)],
        "artifacts": {"permutation_importance": imp_path}
    }

# ----------------- Persistence & artifacts -----------------
def _model_artifact_paths(name: str) -> Tuple[str, str]:
    outdir = make_artifacts_dir()
    return os.path.join(outdir, f"{name}.joblib"), os.path.join(outdir, f"{name}.meta.json")

@app.post("/model/save")
def save_model(name: str = Query("best")):
    if joblib is None:
        raise HTTPException(status_code=500, detail="joblib not available; install joblib to enable persistence.")
    if _BEST_MODEL is None or _BEST_FEATURE_NAMES is None or _BEST_MODEL_NAME is None:
        raise HTTPException(status_code=400, detail="No trained model to save.")
    model_path, meta_path = _model_artifact_paths(name)
    joblib.dump(_BEST_MODEL, model_path)
    meta = {"model_name": _BEST_MODEL_NAME, "feature_names": _BEST_FEATURE_NAMES, "version": APP_VERSION}
    save_json(meta_path, meta)
    return {"saved": {"model": model_path, "meta": meta_path}}

@app.post("/model/load")
def load_model(name: str = Query("best")):
    if joblib is None:
        raise HTTPException(status_code=500, detail="joblib not available; install joblib to enable persistence.")
    model_path, meta_path = _model_artifact_paths(name)
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Saved model not found.")
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    global _BEST_MODEL, _BEST_MODEL_NAME, _BEST_FEATURE_NAMES
    _BEST_MODEL = model
    _BEST_MODEL_NAME = meta.get("model_name", "unknown")
    _BEST_FEATURE_NAMES = meta.get("feature_names", [])
    return {"loaded": {"model": model_path, "meta": meta_path}, "best_model": _BEST_MODEL_NAME}

@app.get("/artifacts/list")
def artifacts_list():
    return {"artifacts": list_artifacts()}

@app.get("/artifacts/{filename}")
def artifacts_download(filename: str = Path(..., description="Exact filename under ./artifacts")):
    path = os.path.join(make_artifacts_dir(), filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, filename=filename)

# ----------------- Minimal HTML UI -----------------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    html = """
<!doctype html>
<html>
<head><meta charset="utf-8"/><title>Microplastics Risk Model UI</title></head>
<body style="font-family:sans-serif;margin:24px">
  <h1>Microplastics Risk Model</h1>
  <h2>/predict_csv</h2>
  <form id="predForm">
    <input type="file" name="file" required />
    <label>Include components</label>
    <select name="include_components"><option>true</option><option>false</option></select>
    <button type="submit">Score CSV</button>
  </form>
  <pre id="predOut" style="white-space:pre-wrap;background:#f6f6f6;padding:12px;border-radius:8px"></pre>

  <h2>Artifacts</h2>
  <button id="listBtn">List</button>
  <ul id="files"></ul>

<script>
async function postPredict() {
  const form = document.getElementById('predForm');
  const data = new FormData(form);
  const url = new URL('/predict_csv', window.location.origin);
  url.searchParams.set('include_components', data.get('include_components') === 'true');
  const res = await fetch(url, { method: 'POST', body: data });
  if (res.ok) {
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'scored.csv';
    a.click();
    document.getElementById('predOut').textContent = 'Downloaded scored.csv';
  } else {
    document.getElementById('predOut').textContent = await res.text();
  }
}
document.getElementById('predForm').addEventListener('submit', (e)=>{e.preventDefault(); postPredict();});
document.getElementById('listBtn').addEventListener('click', async ()=>{
  const r = await fetch('/artifacts/list'); const j = await r.json();
  const ul = document.getElementById('files'); ul.innerHTML='';
  j.artifacts.forEach(f=>{ const li=document.createElement('li'); const a=document.createElement('a'); a.href='/artifacts/'+encodeURIComponent(f); a.innerText=f; li.appendChild(a); ul.appendChild(li); });
});
</script>
</body></html>
    """
    return HTMLResponse(content=html)

# ----------------- CSV helpers & CLI -----------------
def _csv_iter(text: str):
    reader = csv.DictReader(StringIO(text))
    required = set(NUMERIC_COLS + CATEGORICAL_COLS)
    missing = required - set(h.strip() for h in reader.fieldnames or [])
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV missing columns: {', '.join(sorted(missing))}")
    for raw in reader:
        try:
            yield {
                "concentration_p_per_L": float(raw["concentration_p_per_L"]),
                "mean_size_um": float(raw["mean_size_um"]),
                "fraction_lt100um": float(raw["fraction_lt100um"]),
                "polymer": (raw["polymer"] or "").strip(),
                "shape": (raw["shape"] or "").strip().lower(),
                "route": (raw["route"] or "").strip().lower(),
                "ingestion_rate_L_per_day": float(raw["ingestion_rate_L_per_day"]),
                "body_weight_kg": float(raw["body_weight_kg"]),
                "exposure_days": int(raw.get("exposure_days") or 365),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

def _csv_to_df(text: str) -> pd.DataFrame:
    rows = list(_csv_iter(text))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", type=str, help="Path to CSV to analyze.")
    parser.add_argument("--scaling", type=str, default="standard", choices=["standard","minmax","none"])
    parser.add_argument("--outliers", type=str, default="winsorize", choices=["winsorize","remove","none"])
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument("--skew", type=str, default="auto", choices=["auto","none"])
    args = parser.parse_args()
    if args.analyze:
        # Minimal CLI run generates artifacts; kept for parity.
        text = open(args.analyze, "r", encoding="utf-8").read()
        df = _csv_to_df(text)
        df = add_risk_columns(df)
        df, wins, removed = handle_outliers(df, NUMERIC_COLS, args.outliers, float(args.iqr_k))
        df, skewness, transformed = transform_skew(df, ["concentration_p_per_L","mean_size_um"], args.skew)
        df_e = encode_categoricals_pd(df, CATEGORICAL_COLS)
        df_s, stats = scale_features_pd(df_e, NUMERIC_COLS, args.scaling)
        outdir = make_artifacts_dir()
        df_s.to_csv(os.path.join(outdir, "processed.csv"), index=False)
        plot_risk_distribution(df_s, outdir)
        plot_risk_by_level(df_s, outdir)
        plot_risk_vs_conc(df_s, outdir)
        plot_polymer_distribution(df, outdir)
        print("Artifacts in ./artifacts")

# ===== File: requirements.txt =====
fastapi==0.114.2
uvicorn[standard]==0.30.6
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
imbalanced-learn==0.12.2
joblib==1.4.2
matplotlib==3.8.4
python-multipart==0.0.9

# ===== File: Dockerfile =====
# Why: slim base + wheels keep image small and fast to build.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (certs, basic build tools for safety)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
