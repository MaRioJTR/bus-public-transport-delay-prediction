from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.transport_delay.pipeline import run_pipeline
from src.transport_delay.explainability import save_shap_outputs


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

    save_shap_outputs(df, output_dir)

    print("Saved explainability outputs:")
    print(output_dir / "shap_mean_abs.csv")
    print(output_dir / "figures" / "shap_summary.png")


if __name__ == "__main__":
    main()
