from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from .cleaning import CleaningConfig, clean_raw_dataset
from .features import add_features, impute_delay_and_actual_time
from .modeling import train_and_evaluate


@dataclass(frozen=True)
class PipelineOutputs:
    cleaned: pd.DataFrame
    model_metrics: pd.DataFrame
    feature_importance: pd.DataFrame


def run_pipeline(input_csv: str | Path, output_dir: str | Path, cfg: CleaningConfig | None = None) -> PipelineOutputs:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv)
    cleaned = clean_raw_dataset(raw, cfg=cfg)
    featured = add_features(cleaned)
    featured = impute_delay_and_actual_time(featured)

    modeling_out: Dict[str, pd.DataFrame] = train_and_evaluate(featured)

    export_cols = [
        "route_id",
        "scheduled_time_iso",
        "actual_time_iso",
        "weather",
        "passenger_count",
        "latitude",
        "longitude",
        "delay_minutes",
        "time_of_day",
        "day_type",
        "weather_severity",
        "route_frequency",
        "scheduled_hour",
        "scheduled_dayofweek",
    ]

    featured[export_cols].to_csv(output_dir / "cleaned_transport_dataset.csv", index=False)
    modeling_out["metrics"].to_csv(output_dir / "model_metrics.csv", index=False)
    modeling_out["feature_importance"].to_csv(output_dir / "feature_importance.csv", index=False)

    return PipelineOutputs(
        cleaned=featured,
        model_metrics=modeling_out["metrics"],
        feature_importance=modeling_out["feature_importance"],
    )
