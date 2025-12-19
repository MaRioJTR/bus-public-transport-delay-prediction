from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["scheduled_hour"] = out["scheduled_dt"].dt.hour
    out["scheduled_dayofweek"] = out["scheduled_dt"].dt.dayofweek
    out["day_type"] = np.where(out["scheduled_dayofweek"] >= 5, "weekend", "weekday")

    def _tod(h: float) -> str:
        if 5 <= h <= 11:
            return "morning"
        if 12 <= h <= 16:
            return "afternoon"
        if 17 <= h <= 21:
            return "evening"
        return "night"

    out["time_of_day"] = out["scheduled_hour"].apply(_tod)

    severity_map = {"sunny": 1, "cloudy": 2, "rainy": 3}
    out["weather_severity"] = out["weather"].map(severity_map).fillna(2).astype(int)

    out["route_frequency"] = out.groupby("route_id")["route_id"].transform("size")

    out["delay_minutes"] = (out["actual_dt"] - out["scheduled_dt"]).dt.total_seconds() / 60.0

    return out


def impute_delay_and_actual_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    delay = out["delay_minutes"].copy()

    group_cols = ["route_id", "time_of_day", "day_type"]
    group_median = out.groupby(group_cols)["delay_minutes"].median()
    global_median = out["delay_minutes"].median()

    def _fill_delay(r: pd.Series) -> float:
        if pd.notna(r["delay_minutes"]):
            return float(r["delay_minutes"])
        key = (r["route_id"], r["time_of_day"], r["day_type"])
        v = group_median.get(key, np.nan)
        if pd.isna(v):
            v = global_median
        return float(v) if pd.notna(v) else 0.0

    out["delay_minutes"] = out.apply(_fill_delay, axis=1)

    q1 = out["delay_minutes"].quantile(0.25)
    q3 = out["delay_minutes"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out["delay_minutes"] = out["delay_minutes"].clip(lower=lower, upper=upper)

    out["actual_dt_filled"] = out["scheduled_dt"] + pd.to_timedelta(out["delay_minutes"], unit="m")
    out["scheduled_time_iso"] = out["scheduled_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["actual_time_iso"] = out["actual_dt_filled"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return out
