from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_eda_figures(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["delay_minutes"].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title("Delay Minutes Distribution")
    ax.set_xlabel("Delay (minutes)")
    fig.tight_layout()
    fig.savefig(output_dir / "delay_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(data=df, x="weather", y="delay_minutes", ax=ax)
    ax.set_title("Delay by Weather")
    fig.tight_layout()
    fig.savefig(output_dir / "delay_by_weather.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    order = ["morning", "afternoon", "evening", "night"]
    sns.boxplot(data=df, x="time_of_day", y="delay_minutes", order=order, ax=ax)
    ax.set_title("Delay by Time of Day")
    fig.tight_layout()
    fig.savefig(output_dir / "delay_by_time_of_day.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(data=df, x="day_type", y="delay_minutes", ax=ax)
    ax.set_title("Delay by Day Type")
    fig.tight_layout()
    fig.savefig(output_dir / "delay_by_day_type.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="route_id", y="passenger_count", ax=ax)
    ax.set_title("Passenger Count by Route (Outlier View)")
    fig.tight_layout()
    fig.savefig(output_dir / "passenger_by_route.png", dpi=150)
    plt.close(fig)

    route_counts = df["route_id"].value_counts().reset_index()
    route_counts.columns = ["route_id", "count"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(data=route_counts, x="route_id", y="count", ax=ax)
    ax.set_title("Route Frequency")
    fig.tight_layout()
    fig.savefig(output_dir / "route_frequency.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=df, x="longitude", y="latitude", hue="route_id", alpha=0.7, ax=ax)
    ax.set_title("GPS Points (After Cleaning)")
    fig.tight_layout()
    fig.savefig(output_dir / "gps_scatter.png", dpi=150)
    plt.close(fig)
