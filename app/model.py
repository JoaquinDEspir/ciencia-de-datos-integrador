"""
Training utilities for the Mendoza price estimator used in the Streamlit app.
Versión: Multi-modelo por tipología + Ridge log-target con uplift por dormitorios (+5%/dorm) + CAP 500k USD
Con manejo robusto de columnas duplicadas y datasets pequeños.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# ================== Config y columnas ==================

TARGET_COLUMN = "precio_usd"
GEO_ZONE_COLUMN = "zona_geografica"
PRICE_CAP_USD = 500_000.0
RANDOM_STATE = 42
MIN_TYPE_SAMPLES = 10
MIN_SPLIT_SAMPLES = 10

NUMERIC_BASE_FEATURES = [
    "superficie_total",
    "superficie_cubierta",
    "banos",
    "cocheras_total",
    "prp_lat",
    "prp_lng",
]

CATEGORICAL_FEATURES = [
    "operacion",
    "tipo_propiedad",
    "loc_desc",
    GEO_ZONE_COLUMN,
]

DERIVED_FEATURES = [
    "log_sup_tot",
    "log_sup_cub",
    "semi",
    "sup_cub_por_bano",
]

MODEL_TYPES_USING_PPM2 = {"Casa", "Departamento", "Depto"}

# ================== Artefactos ==================

@dataclass(frozen=True)
class ModelArtifacts:
    model: "MultiTypePriceModel"
    metrics: Dict[str, float]
    feature_options: Dict[str, List[str]]
    price_summary: Dict[str, float]

# ================== Utilidades ==================

def _safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=40)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _clip_price(a: Union[np.ndarray, pd.Series], cap: float = PRICE_CAP_USD) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=float), 0.0, cap)

def _add_geo_cluster(df: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    d = df.copy()
    if "prp_lat" not in d.columns or "prp_lng" not in d.columns:
        d["geo_cluster"] = -1
        return d
    mask = d["prp_lat"].notna() & d["prp_lng"].notna()
    if mask.sum() < max(k, 3):
        d["geo_cluster"] = -1
        return d
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    d.loc[mask, "geo_cluster"] = km.fit_predict(d.loc[mask, ["prp_lat", "prp_lng"]])
    d["geo_cluster"] = d["geo_cluster"].fillna(-1).astype(int).astype(str)
    return d

def _saneo_por_tipo(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "tipo_propiedad" in d.columns:
        casas = d["tipo_propiedad"].str.contains("Casa", case=False, na=False)
        d = d[~casas | ((d["superficie_cubierta"].between(30, 600)) & (d["superficie_total"].between(50, 2000)))]
        deptos = d["tipo_propiedad"].str.contains("Depto|Departamento", case=False, na=False)
        d = d[~deptos | (d["superficie_cubierta"].between(20, 300))]
    return d

# ================== Ridge log-target + uplift ==================

class RidgeMonotonicPriceModel:
    """Ridge log-linear model + monotonic uplift for bedrooms (+ CAP USD)."""

    def __init__(
        self,
        numeric_base_cols: List[str],
        categorical_cols: List[str],
        *,
        alphas: Optional[np.ndarray] = None,
        smoothing: float = 10.0,
        clip_quantile: float = 0.995,
        min_uplift_step: float = 1.05,
        uplift_cap: float = 2.0,
    ):
        self.numeric_base_cols = list(numeric_base_cols)
        self.categorical_cols = list(categorical_cols)
        self.derived_cols = list(DERIVED_FEATURES)
        self.alphas = alphas if alphas is not None else np.logspace(-3, 3, 13)
        self.smoothing = smoothing
        self.clip_quantile = clip_quantile
        self.min_uplift_step = min_uplift_step
        self.uplift_cap = uplift_cap
        self._reset_prediction_notes()

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        X, y = self._prepare_training_data(df, target_col)
        if X.empty:
            raise RuntimeError("No hay datos suficientes para entrenar el modelo.")

        use_split = len(X) >= MIN_SPLIT_SAMPLES
        pipe = self._build_pipeline(self.numeric_used_, self.categorical_cols)

        if not use_split:
            y_log = np.log1p(y.values)
            pipe.fit(X, y_log)
            base = _clip_price(np.expm1(pipe.predict(X)))
            self.cap_ = float(np.quantile(base, 0.99)) if len(base) > 1 else PRICE_CAP_USD
            self.uplift_by_bedroom_ = self._learn_bedroom_uplift(pipe, X, y, self.cap_, self.min_uplift_step, self.uplift_cap)
            self.pipeline_ = pipe
            return

        try:
            bins = pd.qcut(y, q=min(10, max(2, len(y))), labels=False, duplicates="drop")
            Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=bins, random_state=RANDOM_STATE)
        except ValueError:
            Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=None, random_state=RANDOM_STATE)

        ytr_log = np.log1p(ytr.values)
        Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]
        pipe.fit(Xtr, ytr_log)

        base = _clip_price(np.expm1(pipe.predict(Xtr)))
        self.cap_ = float(np.quantile(base, self.clip_quantile)) if len(base) > 1 else PRICE_CAP_USD
        self.uplift_by_bedroom_ = self._learn_bedroom_uplift(pipe, Xtr, ytr, self.cap_, self.min_uplift_step, self.uplift_cap)
        pipe.fit(X.loc[:, ~X.columns.duplicated()], np.log1p(y.values))
        self.pipeline_ = pipe

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._reset_prediction_notes()
        feats = self._add_features(self._ensure_geo_columns(df))
        raw_base = np.maximum(np.expm1(self.pipeline_.predict(feats)), 0.0)
        base_pred = _clip_price(raw_base)
        uplift = np.ones(len(base_pred))
        if hasattr(self, "uplift_by_bedroom_") and "dormitorios" in feats.columns:
            d = pd.to_numeric(feats["dormitorios"], errors="coerce").round()
            uplift = self._lookup_uplift(d)
        raw_pred = raw_base * uplift
        self._record_high_value_warning(raw_pred)
        return _clip_price(base_pred * uplift)

    def _prepare_training_data(self, df: pd.DataFrame, target_col: str):
        w = df.copy()
        w = w.loc[:, ~w.columns.duplicated()].copy()
        w = w[w[target_col] > 0].copy()
        w[target_col] = _clip_price(w[target_col])
        w = self._ensure_geo_columns(w)
        self.categorical_cols = list(dict.fromkeys(self.categorical_cols + ["geo_cluster"]))
        w = self._add_features(w)
        w = w.loc[:, ~w.columns.duplicated()].copy()
        self.numeric_used_ = [c for c in self.numeric_base_cols + self.derived_cols if c not in self.categorical_cols]
        cat_cols = [c for c in self.categorical_cols if c not in self.numeric_used_]
        X = w[cat_cols + self.numeric_used_ + ["dormitorios"]].copy()
        y = w[target_col].astype(float)
        return X, y

    def _ensure_geo_columns(self, d: pd.DataFrame) -> pd.DataFrame:
        w = d.copy()
        if GEO_ZONE_COLUMN not in w.columns:
            w[GEO_ZONE_COLUMN] = "desconocida"
        w = _add_geo_cluster(w)
        geo = w.get("geo_cluster")
        if geo is None:
            w["geo_cluster"] = "-1"
        else:
            if np.issubdtype(geo.dtype, np.number):
                w["geo_cluster"] = geo.fillna(-1).astype(int).astype(str)
            else:
                w["geo_cluster"] = geo.fillna("-1").astype(str)
        return w

    def _add_features(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for c in ["superficie_total", "superficie_cubierta", "banos", "cocheras_total", "prp_lat", "prp_lng", "dormitorios"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d["log_sup_tot"] = np.log1p(d["superficie_total"])
        d["log_sup_cub"] = np.log1p(d["superficie_cubierta"])
        d["semi"] = d["superficie_total"] - d["superficie_cubierta"]
        with np.errstate(divide="ignore", invalid="ignore"):
            d["sup_cub_por_bano"] = d["superficie_cubierta"] / d["banos"].replace(0, np.nan)
        return d.replace([np.inf, -np.inf], np.nan)

    def _build_pipeline(self, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
        num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        ohe = _safe_ohe()
        cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)])
        pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)], remainder="drop")
        reg = RidgeCV(alphas=self.alphas, cv=5, fit_intercept=True)
        return Pipeline([("prep", pre), ("reg", reg)])

    def _learn_bedroom_uplift(self, pipe, X, y, cap, min_step, uplift_cap):
        if "dormitorios" not in X.columns:
            return {}
        Xlin = X.loc[:, ~X.columns.duplicated()]
        base = _clip_price(np.expm1(pipe.predict(Xlin)))
        tmp = pd.DataFrame({
            "y": y.values,
            "pred": base,
            "dorm": pd.to_numeric(X["dormitorios"], errors="coerce").round()
        }).dropna()
        if tmp.empty:
            return {}
        tmp["ratio"] = tmp["y"] / tmp["pred"]
        g = tmp.groupby("dorm")["ratio"].median().fillna(1)
        uplift = np.clip(np.maximum.accumulate(g.values), 1.0, uplift_cap)
        return dict(zip(g.index.astype(int), uplift))

    def _lookup_uplift(self, dorms):
        if not hasattr(self, "uplift_by_bedroom_"):
            return np.ones(len(dorms))
        return np.array([self.uplift_by_bedroom_.get(int(v), 1.0) for v in dorms.fillna(0)], dtype=float)

    @property
    def last_prediction_notes(self) -> List[str]:
        return getattr(self, "_last_prediction_notes", [])

    @property
    def has_high_value_prediction(self) -> bool:
        mask = getattr(self, "_last_high_value_mask", np.zeros(0, dtype=bool))
        return bool(mask.size and mask.any())

    def _reset_prediction_notes(self) -> None:
        self._last_prediction_notes = []
        self._last_high_value_mask = np.zeros(0, dtype=bool)

    def _record_high_value_warning(self, raw_pred: np.ndarray) -> None:
        mask = np.asarray(raw_pred, dtype=float) > PRICE_CAP_USD
        self._last_high_value_mask = mask
        if not mask.any():
            self._last_prediction_notes = []
            return
        count = int(mask.sum())
        if count == 1:
            note = (
                "La estimación supera los 500k USD. "
                "Para montos altos considera consultar a un profesional."
            )
        else:
            note = (
                f"{count} estimaciones superan los 500k USD. "
                "Para montos altos considera consultar a un profesional."
            )
        self._last_prediction_notes = [note]
        warnings.warn(note, UserWarning, stacklevel=3)

# ================== Variante log(precio/m²) ==================

class RidgeLogPricePerM2Model(RidgeMonotonicPriceModel):
    def fit(self, df: pd.DataFrame, target_col: str):
        df2 = df.copy()
        df2["pp_m2"] = df2[target_col] / df2["superficie_cubierta"].replace(0, np.nan)
        df2 = df2.dropna(subset=["pp_m2"])
        super().fit(df2, target_col="pp_m2")

    def predict(self, df: pd.DataFrame):
        self._reset_prediction_notes()
        feats = self._add_features(self._ensure_geo_columns(df))
        ppm2 = np.maximum(np.expm1(self.pipeline_.predict(feats)), 0.0)
        sup_cub = feats["superficie_cubierta"].fillna(0).to_numpy()
        raw_base = np.maximum(ppm2 * sup_cub, 0.0)
        base = _clip_price(raw_base)
        uplift = np.ones(len(base))
        if hasattr(self, "uplift_by_bedroom_") and "dormitorios" in feats.columns:
            d = pd.to_numeric(feats["dormitorios"], errors="coerce").round()
            uplift = self._lookup_uplift(d)
        raw_pred = raw_base * uplift
        self._record_high_value_warning(raw_pred)
        return _clip_price(base * uplift)

# ================== Multi-modelo por tipo ==================

class MultiTypePriceModel:
    def __init__(self, models_by_type: Dict[str, RidgeMonotonicPriceModel], fallback_model: RidgeMonotonicPriceModel):
        self.models_by_type = models_by_type
        self.fallback_model = fallback_model
        self._last_prediction_notes: List[str] = []

    def predict(self, df: pd.DataFrame):
        self._last_prediction_notes = []
        if "tipo_propiedad" not in df.columns:
            preds = self.fallback_model.predict(df)
            self._collect_notes(self.fallback_model)
            return preds
        preds = np.zeros(len(df))
        tipos = df["tipo_propiedad"].astype(str).fillna("")
        for t in tipos.unique():
            sub = df[tipos == t]
            model = self.models_by_type.get(t, self.fallback_model)
            preds[tipos == t] = model.predict(sub)
            self._collect_notes(model)
        return _clip_price(preds)

    @property
    def last_prediction_notes(self) -> List[str]:
        return list(self._last_prediction_notes)

    def _collect_notes(self, model: RidgeMonotonicPriceModel) -> None:
        notes = getattr(model, "last_prediction_notes", [])
        for note in notes:
            if note not in self._last_prediction_notes:
                self._last_prediction_notes.append(note)

# ================== Entrenamiento principal ==================

def train_price_model(dataset: pd.DataFrame) -> ModelArtifacts:
    required = [TARGET_COLUMN, "dormitorios"] + NUMERIC_BASE_FEATURES + ["operacion", "tipo_propiedad", "loc_desc"]
    miss = [c for c in required if c not in dataset.columns]
    if miss:
        raise RuntimeError(f"Faltan columnas: {miss}")

    df = dataset.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if GEO_ZONE_COLUMN not in df.columns:
        df[GEO_ZONE_COLUMN] = "desconocida"
    df = df[df[TARGET_COLUMN] > 0].copy()
    df[TARGET_COLUMN] = _clip_price(df[TARGET_COLUMN])
    df = _saneo_por_tipo(df)

    # Split global para métricas
    if len(df) >= MIN_SPLIT_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        cut = int(len(idx) * 0.8)
        train_df, test_df = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    else:
        train_df, test_df = df, pd.DataFrame(columns=df.columns)

    models_by_type = {}
    for t in sorted(train_df["tipo_propiedad"].dropna().unique()):
        sub = train_df[train_df["tipo_propiedad"] == t]
        if len(sub) < MIN_TYPE_SAMPLES:
            continue
        uses_ppm2 = any(k.lower() in t.lower() for k in MODEL_TYPES_USING_PPM2)
        cls = RidgeLogPricePerM2Model if uses_ppm2 else RidgeMonotonicPriceModel
        m = cls(NUMERIC_BASE_FEATURES, CATEGORICAL_FEATURES)
        m.fit(sub, TARGET_COLUMN)
        models_by_type[t] = m

    fallback = RidgeMonotonicPriceModel(NUMERIC_BASE_FEATURES, CATEGORICAL_FEATURES)
    fallback.fit(train_df, TARGET_COLUMN)
    wrapper = MultiTypePriceModel(models_by_type, fallback)

    if not test_df.empty:
        y_true = test_df[TARGET_COLUMN].values
        y_pred = wrapper.predict(test_df)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
    else:
        mae = rmse = r2 = float("nan")

    # Reentrenar final
    models_final = {}
    for t in sorted(df["tipo_propiedad"].dropna().unique()):
        sub = df[df["tipo_propiedad"] == t]
        if len(sub) < MIN_TYPE_SAMPLES:
            continue
        uses_ppm2 = any(k.lower() in t.lower() for k in MODEL_TYPES_USING_PPM2)
        cls = RidgeLogPricePerM2Model if uses_ppm2 else RidgeMonotonicPriceModel
        m = cls(NUMERIC_BASE_FEATURES, CATEGORICAL_FEATURES)
        m.fit(sub, TARGET_COLUMN)
        models_final[t] = m

    fallback_final = RidgeMonotonicPriceModel(NUMERIC_BASE_FEATURES, CATEGORICAL_FEATURES)
    fallback_final.fit(df, TARGET_COLUMN)
    final_wrapper = MultiTypePriceModel(models_final, fallback_final)

    feature_options = {
        "operacion": sorted(df["operacion"].dropna().unique()),
        "tipo_propiedad": sorted(df["tipo_propiedad"].dropna().unique()),
        "loc_desc": sorted(df["loc_desc"].dropna().unique()),
        GEO_ZONE_COLUMN: sorted(df[GEO_ZONE_COLUMN].dropna().unique()),
    }

    price_summary = {
        "min": float(df[TARGET_COLUMN].min()),
        "p50": float(df[TARGET_COLUMN].median()),
        "p90": float(df[TARGET_COLUMN].quantile(0.9)),
        "max": float(df[TARGET_COLUMN].max()),
    }

    metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
    return ModelArtifacts(final_wrapper, metrics, feature_options, price_summary)
