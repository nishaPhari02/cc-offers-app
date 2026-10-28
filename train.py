"""Explicit training command: grouped tuning, evaluation and explanations."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from data import (
    DATA_PATH, PROJECT_DIR, FEATURES, FIELD_MAP, LABELS, TARGET,
    GROUP_COL, SEED, PIPELINE_VERSION, dataset_hash, load_data, partition_data,
)
from models import build_models, PARAMETER_GRID
from evaluation import score_predictions, save_model_evaluation
from run_metadata import package_versions, training_code_hash


def train_all(data_path=DATA_PATH, output_dir=PROJECT_DIR, overwrite=False):
    data_path, output_dir = Path(data_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "models.joblib").exists() and not overwrite:
        raise FileExistsError("Models already exist. Use --overwrite to explicitly replace this run.")

    input_hash = dataset_hash(data_path)
    frame = load_data(data_path)
    partitions, demo, manifest, cv = partition_data(frame)
    train, validation, test = (partitions[name] for name in ["training", "validation", "testing"])
    print(f"[data] {len(frame)} real-data candidates, {frame.attrs['duplicates_removed']} repeated rows removed", flush=True)
    print("[split] " + ", ".join(f"{name}={len(part['X'])}" for name, part in partitions.items()), flush=True)
    print(f"[split] {int((manifest['Partition'] == 'demo').sum())} demo-group row(s); no shared groups", flush=True)

    # Stage a complete run; failed training leaves the previous run untouched.
    with TemporaryDirectory(prefix=".training-", dir=output_dir) as stage_name:
        stage = Path(stage_name)
        results = stage / "results"
        results.mkdir()
        manifest.to_csv(results / "split_manifest.csv", index=False)
        demo.to_csv(results / "demo_sample.csv", index=False)
        fold_manifest = []
        for fold, (fit_pos, score_pos) in enumerate(cv, start=1):
            for role, positions in [("fit", fit_pos), ("score", score_pos)]:
                fold_manifest.extend({
                    "fold": fold, "role": role,
                    "dataset_row_index": int(train["X"].index[pos]),
                    GROUP_COL: train["groups"].iloc[pos],
                } for pos in positions)
        pd.DataFrame(fold_manifest).to_csv(results / "tuning_folds.csv", index=False)

        fitted, metric_rows, tuning = {}, [], {}
        tuning_dir = results / "tuning"
        tuning_dir.mkdir()
        for name, model in build_models().items():
            print(f"[tuning] {name}: 8 combinations, {len(cv)} grouped CV folds", flush=True)
            search = GridSearchCV(
                model, PARAMETER_GRID, scoring="f1_macro", cv=cv,
                n_jobs=2, refit=True, error_score="raise",
            )
            search.fit(train["X"], train["y"])
            fitted[name] = search.best_estimator_
            val_prediction = fitted[name].predict(validation["X"])
            metric_rows.append({
                "model": name,
                "validation_f1_macro": float(f1_score(
                    validation["y"], val_prediction, labels=[0, 1, 2],
                    average="macro", zero_division=0,
                )),
            })
            tuning[name] = {
                "best_parameters": search.best_params_,
                "cv_macro_f1": float(search.best_score_), "cv_folds": len(cv),
            }
            slug = name.lower().replace(" ", "_")
            pd.DataFrame(search.cv_results_).to_csv(tuning_dir / f"{slug}_search.csv", index=False)
            (tuning_dir / f"{slug}_best.json").write_text(json.dumps(tuning[name], indent=2), encoding="utf-8")

        # Freeze model selection using unrounded validation scores BEFORE testing.
        best = max(metric_rows, key=lambda row: row["validation_f1_macro"])["model"]
        print(f"[selection] {best}, selected only by validation macro F1", flush=True)
        for row in metric_rows:
            name = row["model"]
            prediction = save_model_evaluation(name, fitted[name], validation, test, results)
            row.update(score_predictions(test["y"], prediction))
        table = pd.DataFrame(metric_rows).sort_values(
            "validation_f1_macro", ascending=False, kind="stable"
        ).reset_index(drop=True)
        table.to_csv(results / "model_comparison.csv", index=False)
        print(table.round(4).to_string(index=False), flush=True)

        # A reference benchmark, never a candidate for serving or model selection.
        baseline = DummyClassifier(strategy="most_frequent").fit(train["X"], train["y"])
        baseline_metrics = score_predictions(test["y"], baseline.predict(test["X"]))
        (results / "majority_baseline.json").write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")

        from eda import generate_eda
        from shap_explanations import save_shap_outputs
        from lime_explanations import save_lime_outputs
        generate_eda(data_path, results / "eda")
        save_shap_outputs(fitted, validation["X"], validation["y"], FEATURES, LABELS, results)
        save_lime_outputs(fitted, train["X"], validation["X"], validation["y"], FEATURES, LABELS, results)

        holdout = {
            field: ("" if pd.isna(demo[column].iloc[0]) else str(demo[column].iloc[0]))
            for field, column in FIELD_MAP.items()
        }
        holdout.update(actual_tier=demo[TARGET].iloc[0], actual_discount=float(demo["Discount Percentage"].iloc[0]))
        metadata = {
            "pipeline_version": PIPELINE_VERSION,
            "dataset_sha256": input_hash,
            "training_code_sha256": training_code_hash(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "versions": package_versions(),
            "dataset_rows": len(frame), "features": FEATURES, "labels": LABELS,
            "class_counts": {label: int((frame[TARGET] == label).sum()) for label in LABELS},
            "partition_counts": manifest["Partition"].value_counts().to_dict(),
            "class_counts_by_partition": {
                name: {label: int((part["y"] == i).sum()) for i, label in enumerate(LABELS)}
                for name, part in partitions.items()
            },
            "selection_metric": "validation_f1_macro", "best": best,
            "tuning": tuning, "fixed_n_estimators": 300, "seed": SEED,
            "dataset_status": "Source-linked collection draft; full eligibility/conditions not verified for every row",
            "evaluation_status": "Development evaluation; test results were already inspected in the supplied run",
            "limitations": [
                "Few Low test examples; per-class estimates are unstable.",
                "Suggested groups require campaign-level review.",
                "Ordinal encoding imposes an arbitrary order on nominal categories.",
                "SHAP and LIME describe the model, not causal effects.",
                "LIME perturbations can include implausible merchant/category combinations.",
            ],
        }
        options = {
            key: sorted(frame[column].dropna().unique().tolist())
            for key, column in zip(
                ["banks", "tiers", "merchants", "categories", "networks"], FEATURES
            )
        }
        bundle = {
            "pipeline_version": PIPELINE_VERSION, "dataset_sha256": input_hash,
            "models": fitted, "best": best, "metrics": table.to_dict("records"),
            "holdout": holdout, "options": options,
        }
        metadata_text = json.dumps(metadata, indent=2, ensure_ascii=False)
        (stage / "model_metadata.json").write_text(metadata_text, encoding="utf-8")
        (results / "run_metadata.json").write_text(metadata_text, encoding="utf-8")
        joblib.dump(bundle, stage / "models.joblib")
        if dataset_hash(data_path) != input_hash:
            raise RuntimeError("Dataset changed during training. No run was committed.")
        existing_results = output_dir / "results"
        if existing_results.exists():
            shutil.rmtree(existing_results)
        shutil.move(str(results), str(existing_results))
        for name in ["models.joblib", "model_metadata.json"]:
            (stage / name).replace(output_dir / name)
    print(f"[done] Saved model, metadata and results to {output_dir}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace the generated current run after successful training")
    args = parser.parse_args()
    train_all(args.data, args.output, args.overwrite)


if __name__ == "__main__":
    main()
