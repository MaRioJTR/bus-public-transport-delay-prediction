from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .modeling import _build_preprocessor, _get_feature_names


def compute_shap_for_random_forest(
    df: pd.DataFrame,
    target: str = "delay_minutes",
    random_state: int = 42,
    max_background: int = 200,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    categorical = ["route_id", "weather", "time_of_day", "day_type"]
    numeric = ["scheduled_hour", "scheduled_dayofweek", "passenger_count", "latitude", "longitude", "weather_severity", "route_frequency"]

    X = df[categorical + numeric].copy()
    y = df[target].astype(float)

    pre = _build_preprocessor(categorical, numeric)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)

    fitted_pre: ColumnTransformer = pipe.named_steps["pre"]
    feature_names = _get_feature_names(fitted_pre)

    X_trans = fitted_pre.transform(X)

    if hasattr(X_trans, "toarray"):
        X_dense = X_trans.toarray()
    else:
        X_dense = np.asarray(X_trans)

    bg = X_dense[: min(len(X_dense), max_background)]

    explainer = shap.TreeExplainer(pipe.named_steps["model"], data=bg)
    shap_values = explainer.shap_values(X_dense)

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_mean_abs = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )

    return shap_mean_abs, shap_values, X_dense, feature_names


def save_shap_outputs(
    df: pd.DataFrame,
    output_dir: str | Path,
    target: str = "delay_minutes",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_mean_abs, shap_values, X_dense, feature_names = compute_shap_for_random_forest(df, target=target)

    shap_mean_abs.to_csv(output_dir / "shap_mean_abs.csv", index=False)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, X_dense, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=150)
    plt.close()
