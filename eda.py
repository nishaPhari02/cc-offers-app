"""Raw dataset checks and charts. Running this module does not train a model."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import DATA_PATH, PROJECT_DIR, FEATURES, DISCOUNT_COL, GROUP_COL, LABELS, add_labels


def generate_eda(data_path=DATA_PATH, output_dir=PROJECT_DIR / "results" / "eda"):
    raw = pd.read_csv(data_path)
    required = FEATURES + [DISCOUNT_COL, GROUP_COL, "Source URL"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing EDA columns: {sorted(missing)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    discount = pd.to_numeric(raw[DISCOUNT_COL], errors="coerce")
    valid = discount.between(0, 100)
    labelled = add_labels(raw.assign(**{DISCOUNT_COL: discount.where(valid)}))
    counts = labelled["Discount Tier"].value_counts().reindex(LABELS, fill_value=0)
    summary = pd.Series({
        "Total rows": len(raw),
        "Exact duplicate rows": int(raw.duplicated().sum()),
        "Repeated inputs and discount within same group": int(raw.duplicated(
            subset=FEATURES + [DISCOUNT_COL, GROUP_COL]
        ).sum()),
        "Missing or nonnumeric discounts": int(discount.isna().sum()),
        "Discounts outside 0–100": int((discount.notna() & ~valid).sum()),
        "Missing split groups": int(raw[GROUP_COL].isna().sum()),
        "Missing source URLs": int(raw["Source URL"].isna().sum()),
        "Independent split groups": int(raw[GROUP_COL].nunique()),
    })
    summary.to_csv(output / "data_quality_summary.csv", header=["value"])
    raw.isna().sum().rename("missing_count").to_csv(output / "missing_values.csv")
    raw.describe(include="all").to_csv(output / "data_summary.csv")
    counts.to_csv(output / "class_counts.csv", header=["count"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    counts.plot.bar(ax=axes[0, 0], color="steelblue", rot=0)
    axes[0, 0].set_title("Discount tier counts")
    axes[0, 0].set_ylabel("Offers")
    discount[valid].plot.hist(ax=axes[0, 1], bins=15, color="seagreen")
    axes[0, 1].set_title("Discount percentage distribution")
    axes[0, 1].set_xlabel("Discount percentage")
    for ax, column, color in [
        (axes[1, 0], "Bank", "slateblue"),
        (axes[1, 1], "Category", "darkorange"),
    ]:
        raw[column].fillna("Unknown").value_counts().sort_values().plot.barh(ax=ax, color=color)
        ax.set_title(f"Offers by {column.lower()}")
    fig.tight_layout()
    fig.savefig(output / "eda_overview.png", dpi=160)
    plt.close(fig)
    print("[EDA]\n" + summary.to_string(), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results" / "eda")
    args = parser.parse_args()
    generate_eda(args.data, args.output)
