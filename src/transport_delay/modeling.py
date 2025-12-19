from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModelResults:
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame


def _build_preprocessor(categorical: List[str], numeric: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            (
                "num",
                Pipeline([("scaler", StandardScaler())]),
                numeric,
            ),
        ],
        remainder="drop",
    )


def _get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(list(transformer.get_feature_names_out(cols)))
        else:
            names.extend(list(cols))
    return names


def train_and_evaluate(df: pd.DataFrame, target: str = "delay_minutes", random_state: int = 42) -> Dict[str, Any]:
    categorical = ["route_id", "weather", "time_of_day", "day_type"]
    numeric = ["scheduled_hour", "scheduled_dayofweek", "passenger_count", "latitude", "longitude", "weather_severity", "route_frequency"]

    X = df[categorical + numeric].copy()
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    to_dense = FunctionTransformer(
        lambda x: x.toarray() if hasattr(x, "toarray") else x,
        accept_sparse=True,
    )

    models: List[Tuple[str, Any, bool]] = [
        ("linear_regression", LinearRegression(), False),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=15,
                weights="distance",
            ),
            True,
        ),
        (
            "gradient_boosting",
            GradientBoostingRegressor(
                random_state=random_state,
            ),
            True,
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
            False,
        ),
    ]

    rows = []
    importance_rows = []

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for model_name, estimator, needs_dense in models:
        pre = _build_preprocessor(categorical, numeric)
        steps = [("pre", pre)]
        if needs_dense:
            steps.append(("to_dense", to_dense))
        steps.append(("model", estimator))
        pipe = Pipeline(steps)
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, preds)

        cv_rmse = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error").mean()

        rows.append(
            {
                "model": model_name,
                "MAE": float(mae),
                "MSE": float(mse),
                "RMSE": float(rmse),
                "R2": float(r2),
                "CV_RMSE": float(cv_rmse),
            }
        )

        if model_name in {"random_forest", "gradient_boosting"}:
            fitted_pre: ColumnTransformer = pipe.named_steps["pre"]
            feature_names = _get_feature_names(fitted_pre)
            importances = pipe.named_steps["model"].feature_importances_
            imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
            imp["model"] = model_name
            importance_rows.append(imp)

    metrics = pd.DataFrame(rows).sort_values("RMSE")
    feature_importance = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame(columns=["model", "feature", "importance"])

    return {"metrics": metrics, "feature_importance": feature_importance}
