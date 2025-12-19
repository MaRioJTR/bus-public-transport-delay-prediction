from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.transport_delay.pipeline import run_pipeline
from src.transport_delay.eda import save_eda_figures


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    input_csv = project_dir / "dirty_transport_dataset.csv"
    output_dir = project_dir / "outputs"

    cleaned_csv = output_dir / "cleaned_transport_dataset.csv"

    if not cleaned_csv.exists():
        out = run_pipeline(input_csv=input_csv, output_dir=output_dir)
        df = out.cleaned
    else:
        df = pd.read_csv(cleaned_csv)

    figures_dir = output_dir / "figures"
    save_eda_figures(df, figures_dir)

    print("Saved figures to:")
    print(figures_dir)


if __name__ == "__main__":
    main()
