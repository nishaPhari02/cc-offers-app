"""Common evaluation and permutation plots, generated once per model."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score,
)

from data import FEATURES, LABELS, SEED


def score_predictions(y, prediction):
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1_weighted": float(f1_score(
            y, prediction, labels=[0, 1, 2], average="weighted", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y, prediction, labels=[0, 1, 2], average="macro", zero_division=0
        )),
    }


def save_model_evaluation(name, model, validation, testing, output_dir):
    output = Path(output_dir) / name.lower().replace(" ", "_")
    output.mkdir(parents=True, exist_ok=True)
    prediction = model.predict(testing["X"])
    report = classification_report(
        testing["y"], prediction, labels=[0, 1, 2],
        target_names=LABELS, zero_division=0,
    )
    (output / "classification_report.txt").write_text(report, encoding="utf-8")
    display = ConfusionMatrixDisplay.from_predictions(
        testing["y"], prediction, labels=[0, 1, 2],
        display_labels=LABELS, cmap="Blues", colorbar=False,
    )
    display.ax_.set_title(f"Test-set confusion matrix: {name}")
    display.figure_.tight_layout()
    display.figure_.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)

    importance = permutation_importance(
        model, validation["X"], validation["y"], scoring="f1_macro",
        n_repeats=20, random_state=SEED, n_jobs=1,
    )
    table = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    table.to_csv(output / "feature_importance.csv", index=False)
    plot = table.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(plot["feature"], plot["importance_mean"], xerr=plot["importance_std"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Decrease in validation macro F1 after shuffling")
    ax.set_title(f"Permutation importance: {name}")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=160)
    plt.close(fig)
    return prediction
