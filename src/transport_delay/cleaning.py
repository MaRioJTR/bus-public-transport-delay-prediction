from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from dateutil import parser


@dataclass(frozen=True)
class CleaningConfig:
    passenger_min: int = 1
    passenger_max: int = 200
    delay_clip_iqr_multiplier: float = 1.5
    midnight_missing_threshold_hours: int = 6


def _normalize_route_id(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    n = int(m.group(1))
    return f"R{n:02d}"


def _normalize_weather(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)
    if s in {"sun", "sunny"}:
        return "sunny"
    if s in {"cloudy", "clody", "cloudy"}:
        return "cloudy"
    if s in {"rain", "rainy"}:
        return "rainy"
    return s


def _standardize_time_string(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None

    s = s.replace(" ", "")

    m = re.fullmatch(r"(\d{3,4})", s)
    if m:
        digits = m.group(1)
        if len(digits) == 3:
            digits = "0" + digits
        hh = digits[:2]
        mm = digits[2:]
        return f"{hh}:{mm}"

    s = re.sub(r"(?<=\d)\.(?=\d)", ":", s)
    s = re.sub(r"(?i)(am|pm)$", lambda x: x.group(1).upper(), s)
    return s


def _parse_actual_datetime(scheduled_dt: pd.Timestamp, actual_raw: object, cfg: CleaningConfig) -> pd.Timestamp:
    s = _standardize_time_string(actual_raw)
    if s is None:
        return pd.NaT

    try:
        t = parser.parse(s, fuzzy=True).time()
    except Exception:
        return pd.NaT

    candidate = pd.Timestamp(datetime.combine(scheduled_dt.date(), t))

    if candidate.time() == datetime.min.time():
        if abs((scheduled_dt - candidate).total_seconds()) > cfg.midnight_missing_threshold_hours * 3600:
            return pd.NaT

    if candidate < scheduled_dt - pd.Timedelta(hours=cfg.midnight_missing_threshold_hours):
        candidate = candidate + pd.Timedelta(days=1)

    return candidate


def clean_raw_dataset(df: pd.DataFrame, cfg: CleaningConfig | None = None) -> pd.DataFrame:
    cfg = cfg or CleaningConfig()

    out = df.copy()

    out["route_id"] = out["route_id"].apply(_normalize_route_id)
    out["weather"] = out["weather"].apply(_normalize_weather)

    out["scheduled_dt"] = pd.to_datetime(out["scheduled_time"], errors="coerce")

    out["passenger_count"] = pd.to_numeric(out["passenger_count"], errors="coerce")
    out.loc[(out["passenger_count"] < cfg.passenger_min) | (out["passenger_count"] > cfg.passenger_max), "passenger_count"] = np.nan

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out.loc[(out["latitude"].abs() > 90) | (out["longitude"].abs() > 180), ["latitude", "longitude"]] = np.nan

    out["actual_dt"] = out.apply(
        lambda r: _parse_actual_datetime(r["scheduled_dt"], r["actual_time"], cfg) if pd.notna(r["scheduled_dt"]) else pd.NaT,
        axis=1,
    )

    out["weather"] = out["weather"].fillna(out["weather"].mode(dropna=True).iloc[0] if out["weather"].notna().any() else "unknown")

    out["route_id"] = out["route_id"].fillna("R00")

    passenger_median_by_route = out.groupby("route_id")["passenger_count"].median()
    passenger_global_median = out["passenger_count"].median()
    out["passenger_count"] = out.apply(
        lambda r: passenger_median_by_route.get(r["route_id"], passenger_global_median)
        if pd.isna(r["passenger_count"])
        else r["passenger_count"],
        axis=1,
    )

    lat_median_by_route = out.groupby("route_id")["latitude"].median()
    lon_median_by_route = out.groupby("route_id")["longitude"].median()
    lat_global = out["latitude"].median()
    lon_global = out["longitude"].median()

    out["latitude"] = out.apply(
        lambda r: lat_median_by_route.get(r["route_id"], lat_global) if pd.isna(r["latitude"]) else r["latitude"],
        axis=1,
    )
    out["longitude"] = out.apply(
        lambda r: lon_median_by_route.get(r["route_id"], lon_global) if pd.isna(r["longitude"]) else r["longitude"],
        axis=1,
    )

    return out
