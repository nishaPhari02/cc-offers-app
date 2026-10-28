
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_lime_outputs(
    models, X_train, X_val, y_val, features, labels, output_dir
):
    output = Path(output_dir) / "lime"
    output.mkdir(parents=True, exist_ok=True)

    # Same first example used by the SHAP script.
    sample = X_val.sample(
        n=min(100, len(X_val)), random_state=42
    ).iloc[[0]]

    sample.to_csv(output / "explained_input.csv", index=False)

    for name, pipeline in models.items():
        print(f"[LIME] Explaining {name} ...", flush=True)

        prep = pipeline.named_steps["prep"]
        classifier = pipeline.named_steps["clf"]
        encoder = (
            prep.named_transformers_["cat"].named_steps["enc"]
        )

        # Shift encoded categories by one:
        # unknown -1 becomes 0, known categories begin at 1.
        training = (
            np.asarray(prep.transform(X_train), dtype=float) + 1
        )
        row = (
            np.asarray(prep.transform(sample), dtype=float)[0] + 1
        )

        category_names = {
            i: ["Unseen category"] + [str(v) for v in categories]
            for i, categories in enumerate(encoder.categories_)
        }

        class_names = [
            labels[int(class_id)] for class_id in classifier.classes_
        ]

        explainer = LimeTabularExplainer(
            training_data=training,
            mode="classification",
            feature_names=features,
            class_names=class_names,
            categorical_features=list(range(len(features))),
            categorical_names=category_names,
            discretize_continuous=False,
            random_state=42
        )

        def predict_encoded(values):
            # Convert LIME's shifted codes back to model codes.
            return classifier.predict_proba(
                np.asarray(values, dtype=float) - 1
            )

        probabilities = predict_encoded(row.reshape(1, -1))[0]
        pipeline_probabilities = pipeline.predict_proba(sample)[0]

        if not np.allclose(probabilities, pipeline_probabilities):
            raise ValueError("LIME prediction wrapper does not match pipeline.")

        class_position = int(np.argmax(probabilities))

        explanation = explainer.explain_instance(
            data_row=row,
            predict_fn=predict_encoded,
            labels=(class_position,),
            num_features=len(features),
            num_samples=3000
        )

        safe_name = name.lower().replace(" ", "_")

        explanation.save_to_file(
            str(output / f"{safe_name}_explanation.html")
        )

        fig = explanation.as_pyplot_figure(label=class_position)
        fig.tight_layout()
        fig.savefig(
            output / f"{safe_name}_explanation.png",
            dpi=160,
            bbox_inches="tight"
        )
        plt.close(fig)

        pd.DataFrame(
            explanation.as_list(label=class_position),
            columns=["condition", "local_weight"]
        ).to_csv(
            output / f"{safe_name}_weights.csv",
            index=False
        )

        actual = labels[int(y_val.loc[sample.index[0]])]
        predicted = class_names[class_position]

        summary = (
            f"Model: {name}\n"
            f"Actual class: {actual}\n"
            f"Predicted class: {predicted}\n"
            f"Model probability: {probabilities[class_position]:.6f}\n"
            f"Local surrogate R-squared: {explanation.score:.6f}\n"
            f"LIME local estimate: "
            f"{float(explanation.local_pred[0]):.6f}\n"
            "Positive weights support the explained class locally.\n"
            "Negative weights oppose that class locally.\n"
            "LIME is an approximation, not a causal explanation.\n"
            "Perturbed inputs may include unrealistic category combinations.\n"
        )

        (output / f"{safe_name}_summary.txt").write_text(summary, encoding="utf-8")
        metadata = {
            "model": name,
            "dataset_row_index": int(sample.index[0]),
            "actual_class": actual,
            "predicted_class": predicted,
            "model_probability": float(probabilities[class_position]),
            "local_r_squared": float(explanation.score),
            "local_estimate": float(explanation.local_pred[0]),
            "absolute_prediction_gap": float(abs(
                probabilities[class_position] - explanation.local_pred[0]
            )),
            "note": "One validation example; local fit is not model accuracy. Inspect both fit and prediction gap.",
        }
        (output / f"{safe_name}_summary.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    print("[LIME] Outputs saved successfully.", flush=True)
