"""Training utilities for the Mendoza price estimator used in the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

TARGET_COLUMN = "precio_usd"
NUMERIC_FEATURES = [
    "superficie_total",
    "superficie_cubierta",
    "banos",
    "dormitorios",
    "cocheras_total",
    "fotos_cantidad",
    "prp_lat",
    "prp_lng",
]
CATEGORICAL_FEATURES = ["operacion", "tipo_propiedad", "loc_desc"]


@dataclass(frozen=True)
class ModelArtifacts:
    model: "LinearPriceModel"
    metrics: Dict[str, float]
    feature_options: Dict[str, List[str]]
    price_summary: Dict[str, float]


class LinearPriceModel:
    """Minimal linear regressor with log-target training and one-hot features."""

    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]) -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.numeric_impute_: pd.Series | None = None
        self.numeric_mean_: pd.Series | None = None
        self.numeric_std_: pd.Series | None = None
        self.categorical_impute_: Dict[str, str] | None = None
        self.dummy_columns_: List[str] | None = None
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        features = df[self.numeric_cols + self.categorical_cols].copy()
        X = self._prepare_features(features, fit=True)
        y = np.log1p(df[target_col].to_numpy(dtype=float))
        ones = np.ones((X.shape[0], 1), dtype=float)
        X_mat = np.hstack([ones, X.to_numpy(dtype=float)])
        beta, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("The model is not fitted.")
        features = df[self.numeric_cols + self.categorical_cols].copy()
        X = self._prepare_features(features, fit=False)
        log_pred = self.intercept_ + X.to_numpy(dtype=float) @ self.coef_
        return np.clip(np.expm1(log_pred), a_min=0.0, a_max=None)

    def _prepare_features(self, features: pd.DataFrame, fit: bool) -> pd.DataFrame:
        numeric = features[self.numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        if fit:
            self.numeric_impute_ = numeric.median().fillna(0.0)
        numeric = numeric.fillna(self.numeric_impute_)

        if fit:
            self.numeric_mean_ = numeric.mean()
            std = numeric.std().fillna(0.0).replace(0, 1.0)
            self.numeric_std_ = std
        numeric = (numeric - self.numeric_mean_) / self.numeric_std_

        categorical = features[self.categorical_cols].copy()
        categorical = categorical.replace({None: "Desconocido"})
        if fit:
            fill_values = {}
            for col in self.categorical_cols:
                series = categorical[col].astype(str)
                mode = series.mode(dropna=True)
                fill_values[col] = str(mode.iloc[0]) if not mode.empty else "Desconocido"
            self.categorical_impute_ = fill_values
        categorical = categorical.fillna(self.categorical_impute_)
        categorical = categorical.astype(str).replace({"nan": "Desconocido"})
        dummies = pd.get_dummies(categorical, prefix=self.categorical_cols, dtype=float)

        if fit:
            self.dummy_columns_ = dummies.columns.tolist()
        missing_cols = set(self.dummy_columns_ or []) - set(dummies.columns)
        for col in missing_cols:
            dummies[col] = 0.0
        if self.dummy_columns_:
            dummies = dummies[self.dummy_columns_]

        combined = pd.concat([numeric, dummies], axis=1)
        combined = combined.fillna(0.0)
        return combined


def train_price_model(dataset: pd.DataFrame) -> ModelArtifacts:
    working = dataset.copy()
    working = working[working[TARGET_COLUMN] > 0]
    working = working.dropna(subset=[TARGET_COLUMN])
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    working = working[columns].copy()

    if working.empty:
        raise RuntimeError("No hay datos suficientes para entrenar el modelo.")

    rng = np.random.default_rng(seed=42)
    indices = np.arange(len(working))
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    train_df, test_df = working.iloc[train_idx], working.iloc[test_idx]

    base_model = LinearPriceModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    base_model.fit(train_df, TARGET_COLUMN)
    if not test_df.empty:
        predictions = base_model.predict(test_df)
        truth = test_df[TARGET_COLUMN].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(truth - predictions)))
        rmse = float(np.sqrt(np.mean((truth - predictions) ** 2)))
        denom = np.sum((truth - truth.mean()) ** 2)
        r2 = float(1 - np.sum((truth - predictions) ** 2) / denom) if denom else float("nan")
    else:
        mae = rmse = r2 = float("nan")

    final_model = LinearPriceModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    final_model.fit(working, TARGET_COLUMN)

    feature_options = {
        "operacion": sorted(working["operacion"].dropna().unique().tolist()),
        "tipo_propiedad": sorted(working["tipo_propiedad"].dropna().unique().tolist()),
        "loc_desc": sorted(working["loc_desc"].dropna().unique().tolist()),
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
