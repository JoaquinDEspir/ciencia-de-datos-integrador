"""
Training utilities for the Mendoza price estimator used in the Streamlit app.
Versión: Ridge log-target + uplift monótono por dormitorios (con piso +5%/dorm)
Compatible con scikit-learn >= 0.23 y >=1.6
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "precio_usd"
GEO_ZONE_COLUMN = "zona_geografica"

NUMERIC_BASE_FEATURES = [
    "superficie_total",
    "superficie_cubierta",
    "banos",
    "cocheras_total",
    "prp_lat",
    "prp_lng",
]
CATEGORICAL_FEATURES = ["operacion", "tipo_propiedad", "loc_desc", GEO_ZONE_COLUMN]
DERIVED_FEATURES = [
    "log_sup_tot",
    "log_sup_cub",
    "semi",
    "sup_cub_por_bano",
]


@dataclass(frozen=True)
class ModelArtifacts:
    model: "RidgeMonotonicPriceModel"
    metrics: Dict[str, float]
    feature_options: Dict[str, List[str]]
    price_summary: Dict[str, float]


class RidgeMonotonicPriceModel:
    """Ridge log-linear model + monotonic uplift for bedrooms"""

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
        self.smoothing = float(smoothing)
        self.clip_quantile = float(clip_quantile)
        self.min_uplift_step = float(min_uplift_step)
        self.uplift_cap = float(uplift_cap)

        self.pipeline_: Pipeline | None = None
        self.uplift_by_bedroom_: Dict[int, float] | None = None
        self._uplift_min_: int | None = None
        self._uplift_max_: int | None = None
        self.cap_: float | None = None
        self.numeric_used_: List[str] | None = None

    # ========= Public API =========
    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        X, y = self._prepare_training_data(df, target_col)
        if X.empty:
            raise RuntimeError("No hay datos suficientes para entrenar el modelo.")

        price_bins = pd.qcut(y, q=min(10, max(2, len(y))), labels=False, duplicates="drop")
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=price_bins)

        pipe = self._build_pipeline(self.numeric_used_, self.categorical_cols)
        ytr_log = np.log1p(ytr.values.astype(float))
        pipe.fit(Xtr[self.categorical_cols + self.numeric_used_], ytr_log)

        base_tr = np.expm1(pipe.predict(Xtr[self.categorical_cols + self.numeric_used_]))
        self.cap_ = float(np.quantile(base_tr, self.clip_quantile))

        self.uplift_by_bedroom_ = self._learn_bedroom_uplift(
            pipe, Xtr, ytr, self.cap_, self.min_uplift_step, self.uplift_cap
        )
        if self.uplift_by_bedroom_:
            ks = sorted(self.uplift_by_bedroom_.keys())
            self._uplift_min_, self._uplift_max_ = int(ks[0]), int(ks[-1])

        y_all_log = np.log1p(y.values.astype(float))
        pipe.fit(X[self.categorical_cols + self.numeric_used_], y_all_log)
        self.pipeline_ = pipe

    def predict(
        self, df: pd.DataFrame, *, debug: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.pipeline_ is None or self.numeric_used_ is None:
            raise RuntimeError("El modelo no está entrenado.")

        feats = df[self.numeric_base_cols + self.categorical_cols + ["dormitorios"]].copy()
        feats = self._add_features(feats)

        X_lin = feats[self.categorical_cols + self.numeric_used_]
        base_pred = np.expm1(self.pipeline_.predict(X_lin))
        if self.cap_ is not None:
            base_pred = np.clip(base_pred, 0.0, self.cap_)

        uplift = np.ones(len(base_pred), dtype=float)
        if self.uplift_by_bedroom_ and "dormitorios" in feats.columns:
            d = pd.to_numeric(feats["dormitorios"], errors="coerce").round()
            uplift = self._lookup_uplift(d)

        final = base_pred * uplift
        return (final, base_pred, uplift) if debug else final

    # ========= Internals =========
    def _prepare_training_data(self, df: pd.DataFrame, target_col: str):
        w = df.copy()
        w = w[(w[target_col] > 0) & w[target_col].notna()]
        needed = [target_col, "dormitorios"] + self.numeric_base_cols + self.categorical_cols
        miss = [c for c in needed if c not in w.columns]
        if miss:
            raise RuntimeError(f"Faltan columnas requeridas: {miss}")

        w = self._add_features(w[self.numeric_base_cols + self.categorical_cols + ["dormitorios", target_col]].copy())
        self.numeric_used_ = self.numeric_base_cols + self.derived_cols

        X = w[self.categorical_cols + self.numeric_used_ + ["dormitorios"]]
        y = w[target_col].astype(float)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(subset=["superficie_total", "superficie_cubierta", "banos"])
        y = y.loc[X.index]
        return X, y

    def _add_features(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for c in ["superficie_total","superficie_cubierta","banos","cocheras_total","prp_lat","prp_lng","dormitorios"]:
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
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)])
        pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])
        reg = RidgeCV(alphas=self.alphas, cv=5, fit_intercept=True)
        return Pipeline([("prep", pre), ("reg", reg)])

    def _learn_bedroom_uplift(
        self, pipe: Pipeline, Xtr: pd.DataFrame, ytr: pd.Series,
        cap: Optional[float], min_step: float, uplift_cap: float,
    ) -> Dict[int, float]:
        Xlin = Xtr[self.categorical_cols + self.numeric_used_]
        base = np.expm1(pipe.predict(Xlin))
        if cap is not None:
            base = np.clip(base, 0.0, cap)
        tmp = pd.DataFrame({
            "y": ytr.values.astype(float),
            "pred": base,
            "dorm": pd.to_numeric(Xtr["dormitorios"], errors="coerce").round(),
        }).dropna(subset=["dorm"])
        tmp = tmp[tmp["pred"] > 0]
        if tmp.empty:
            return {}
        tmp["dint"] = tmp["dorm"].astype(int)
        g = tmp.groupby("dint").apply(
            lambda g: pd.Series({"median": np.median(g["y"]/g["pred"]), "count": len(g)})
        ).sort_index()
        g["smoothed"] = (g["median"]*g["count"] + 1.0*self.smoothing) / (g["count"] + self.smoothing)
        i0, i1 = int(g.index.min()), int(g.index.max())
        full = pd.RangeIndex(i0, i1+1)
        s = g["smoothed"].reindex(full).ffill().bfill().fillna(1.0).to_numpy()
        mono = np.maximum.accumulate(s)
        base0 = mono[0] if mono.size else 1.0
        uplift = mono / (base0 if base0 > 0 else 1.0)
        uplift = np.clip(uplift, 1.0, uplift_cap)
        for i in range(1, len(uplift)):
            uplift[i] = max(uplift[i], uplift[i-1]*min_step)
            uplift[i] = min(uplift[i], uplift_cap)
        return {int(k): float(v) for k, v in zip(full, uplift)}

    def _lookup_uplift(self, dorm_series: pd.Series) -> np.ndarray:
        if not self.uplift_by_bedroom_ or self._uplift_min_ is None:
            return np.ones(len(dorm_series), dtype=float)
        lo, hi = self._uplift_min_, self._uplift_max_
        vals = pd.to_numeric(dorm_series, errors="coerce").fillna(lo).round().astype(int).clip(lo, hi)
        return np.array([self.uplift_by_bedroom_.get(int(v), 1.0) for v in vals], dtype=float)


def train_price_model(dataset: pd.DataFrame) -> ModelArtifacts:
    required = [TARGET_COLUMN, "dormitorios"] + NUMERIC_BASE_FEATURES + CATEGORICAL_FEATURES
    miss = [c for c in required if c not in dataset.columns]
    if miss:
        raise RuntimeError(f"Faltan columnas para entrenar: {miss}")

    working = dataset.copy()
    working = working[(working[TARGET_COLUMN] > 0) & working[TARGET_COLUMN].notna()]
    rng = np.random.default_rng(42)
    idx = np.arange(len(working))
    rng.shuffle(idx)
    cut = int(len(idx)*0.8)
    train_df, test_df = working.iloc[idx[:cut]], working.iloc[idx[cut:]]

    base_model = RidgeMonotonicPriceModel(
        numeric_base_cols=NUMERIC_BASE_FEATURES,
        categorical_cols=CATEGORICAL_FEATURES,
    )
    base_model.fit(train_df, TARGET_COLUMN)

    if not test_df.empty:
        y_true = test_df[TARGET_COLUMN].to_numpy(dtype=float)
        y_pred = base_model.predict(test_df)
        mae = float(mean_absolute_error(y_true, y_pred))
        try:
            rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        except TypeError:
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
    else:
        mae = rmse = r2 = float("nan")

    final_model = RidgeMonotonicPriceModel(
        numeric_base_cols=NUMERIC_BASE_FEATURES,
        categorical_cols=CATEGORICAL_FEATURES,
    )
    final_model.fit(working, TARGET_COLUMN)

    feature_options = {
        "operacion": sorted(working["operacion"].dropna().astype(str).unique().tolist()),
        "tipo_propiedad": sorted(working["tipo_propiedad"].dropna().astype(str).unique().tolist()),
        "loc_desc": sorted(working["loc_desc"].dropna().astype(str).unique().tolist()),
        GEO_ZONE_COLUMN: sorted(working[GEO_ZONE_COLUMN].dropna().astype(str).unique().tolist()),
    }
    price_summary = {
        "min": float(working[TARGET_COLUMN].min()),
        "p50": float(working[TARGET_COLUMN].median()),
        "p90": float(working[TARGET_COLUMN].quantile(0.9)),
        "max": float(working[TARGET_COLUMN].max()),
    }
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return ModelArtifacts(
        model=final_model,
        metrics=metrics,
        feature_options=feature_options,
        price_summary=price_summary,
    )
