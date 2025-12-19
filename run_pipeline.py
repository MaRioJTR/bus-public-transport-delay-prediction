from __future__ import annotations

from pathlib import Path

from src.transport_delay.pipeline import run_pipeline


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    input_csv = project_dir / "dirty_transport_dataset.csv"
    output_dir = project_dir / "outputs"

    out = run_pipeline(input_csv=input_csv, output_dir=output_dir)

    print("Saved:")
    print(output_dir / "cleaned_transport_dataset.csv")
    print(output_dir / "model_metrics.csv")
    print(output_dir / "feature_importance.csv")
    print("\nModel metrics:")
    print(out.model_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
