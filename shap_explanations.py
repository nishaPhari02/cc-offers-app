
from pathlib import Path
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_shap_outputs(models, X_val, y_val, features, labels, output_dir):
    output = Path(output_dir) / "shap"
    output.mkdir(parents=True, exist_ok=True)

    # A reproducible sample of validation rows.
    sample = X_val.sample(
        n=min(100, len(X_val)), random_state=42
    )
    sample.to_csv(output / "explained_validation_inputs.csv", index=False)

    for name, pipeline in models.items():
        print(f"[SHAP] Explaining {name} ...", flush=True)

        prep = pipeline.named_steps["prep"]
        classifier = pipeline.named_steps["clf"]
        encoded = np.asarray(prep.transform(sample), dtype=float)

        explainer = shap.TreeExplainer(
            classifier,
            feature_perturbation="tree_path_dependent"
        )
        values = explainer.shap_values(encoded)

        # Handle both older and newer SHAP output formats.
        if isinstance(values, list):
            values = np.stack(values, axis=-1)
        values = np.asarray(values)

        expected_shape = (
            len(sample), len(features), len(classifier.classes_)
        )
        if values.shape != expected_shape:
            raise ValueError(
                f"Unexpected SHAP shape: {values.shape}; "
                f"expected {expected_shape}"
            )

        safe_name = name.lower().replace(" ", "_")

        # Global importance: mean absolute SHAP across rows and classes.
        global_table = pd.DataFrame({
            "feature": features,
            "mean_absolute_shap": np.abs(values).mean(axis=(0, 2))
        }).sort_values("mean_absolute_shap")

        global_table.to_csv(
            output / f"{safe_name}_global.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            global_table["feature"],
            global_table["mean_absolute_shap"]
        )
        ax.set_xlabel("Mean absolute SHAP value across rows and classes")
        ax.set_title(f"SHAP global importance: {name}")
        fig.tight_layout()
        fig.savefig(output / f"{safe_name}_global.png", dpi=160)
        plt.close(fig)

        # Explain the predicted class for the first sampled row.
        probabilities = pipeline.predict_proba(sample.iloc[[0]])[0]
        class_position = int(np.argmax(probabilities))
        predicted_label = labels[
            int(classifier.classes_[class_position])
        ]
        base = float(
            np.asarray(explainer.expected_value)[class_position]
        )
        contributions = values[0, :, class_position]

        if not np.isclose(
            base + contributions.sum(),
            probabilities[class_position],
            atol=1e-5
        ):
            raise ValueError("SHAP contributions failed the probability check.")

        local_table = pd.DataFrame({
            "feature": features,
            "input_value": sample.iloc[0].to_numpy(),
            "shap_value": contributions
        })
        local_table.to_csv(
            output / f"{safe_name}_local.csv", index=False
        )

        plot_table = local_table.sort_values("shap_value")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            plot_table["feature"],
            plot_table["shap_value"],
            color=[
                "seagreen" if value >= 0 else "salmon"
                for value in plot_table["shap_value"]
            ]
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            f"Contribution toward predicted class: {predicted_label}"
        )
        ax.set_title(f"SHAP individual explanation: {name}")
        fig.tight_layout()
        fig.savefig(output / f"{safe_name}_local.png", dpi=160)
        plt.close(fig)

        actual_label = labels[int(y_val.loc[sample.index[0]])]
        summary = (
            f"Model: {name}\n"
            f"Actual class: {actual_label}\n"
            f"Predicted class: {predicted_label}\n"
            f"Predicted probability: {probabilities[class_position]:.6f}\n"
            f"SHAP baseline: {base:.6f}\n"
            "Baseline plus contributions equals predicted probability.\n"
            "Explanations describe the model, not causal effects.\n"
        )
        (output / f"{safe_name}_local_summary.txt").write_text(summary, encoding="utf-8")
        metadata = {
            "model": name,
            "dataset_row_index": int(sample.index[0]),
            "actual_class": actual_label,
            "predicted_class": predicted_label,
            "predicted_probability": float(probabilities[class_position]),
            "baseline": base,
            "contribution_sum": float(contributions.sum()),
            "additivity_checked": True,
        }
        (output / f"{safe_name}_summary.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    print("[SHAP] Outputs saved successfully.", flush=True)
