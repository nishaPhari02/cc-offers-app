"""Flask inference server. Training is an explicit, separate command."""

import argparse
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from data import DATA_PATH, PROJECT_DIR, FEATURES, FIELD_MAP, LABELS, dataset_hash
from run_metadata import cache_is_current


class ModelUnavailable(RuntimeError):
    pass


def create_app(data_path=DATA_PATH, model_dir=PROJECT_DIR):
    application = Flask(__name__)
    data_path, model_dir = Path(data_path), Path(model_dir)
    state = {"bundle": None, "token": None}

    def bundle():
        model_path = model_dir / "models.joblib"
        metadata_path = model_dir / "model_metadata.json"
        if not model_path.is_file() or not metadata_path.is_file():
            raise ModelUnavailable("No current model. Run python train.py first.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not cache_is_current(metadata, dataset_hash(data_path)):
                raise ModelUnavailable("Data, training code or scikit-learn changed. Run python train.py --overwrite, then restart the app.")
            token = (model_path.stat().st_mtime_ns, metadata_path.stat().st_mtime_ns)
            if state["bundle"] is None or state["token"] != token:
                loaded = joblib.load(model_path)
                if loaded.get("dataset_sha256") != metadata["dataset_sha256"]:
                    raise ModelUnavailable("Model and metadata do not match. Retrain the model.")
                state.update(bundle=loaded, token=token)
            return state["bundle"], metadata
        except ModelUnavailable:
            raise
        except Exception as error:
            raise ModelUnavailable("Cannot load this model. Run python train.py --overwrite in the current environment.") from error

    @application.errorhandler(ModelUnavailable)
    def unavailable(error):
        return jsonify(error=str(error)), 503

    @application.get("/")
    def index():
        return send_from_directory(PROJECT_DIR, "index.html")

    @application.get("/health")
    def health():
        _, metadata = bundle()
        return jsonify(status="ready", model=metadata["best"], dataset_rows=metadata["dataset_rows"])

    @application.get("/options")
    def options():
        saved, _ = bundle()
        return jsonify(saved["options"])

    @application.get("/metrics")
    def metrics():
        saved, metadata = bundle()
        return jsonify(
            metrics=saved["metrics"], best=saved["best"],
            dataset_rows=metadata["dataset_rows"],
            dataset_status=metadata["dataset_status"],
            partition_counts=metadata["partition_counts"],
        )

    @application.get("/holdout")
    def holdout():
        saved, _ = bundle()
        return jsonify(saved["holdout"])

    @application.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object containing prediction inputs."), 400
        if set(payload) - set(FIELD_MAP):
            return jsonify(error="Only bank, tier, merchant, category and network are accepted."), 400
        values = {}
        for field, column in FIELD_MAP.items():
            value = payload.get(field)
            if value is not None and not isinstance(value, str):
                return jsonify(error=f"{field} must be text or null."), 400
            values[column] = value.strip() if isinstance(value, str) and value.strip() else np.nan
        row = pd.DataFrame([values], columns=FEATURES)
        if row.isna().all(axis=None):
            return jsonify(error="Provide at least one of bank, tier, merchant, category or network."), 400

        saved, _ = bundle()
        predictions = {
            name: LABELS[int(model.predict(row)[0])]
            for name, model in saved["models"].items()
        }
        model = saved["models"][saved["best"]]
        probabilities = {
            LABELS[int(class_id)]: float(probability)
            for class_id, probability in zip(model.classes_, model.predict_proba(row)[0])
        }
        encoder = model.named_steps["prep"].named_transformers_["cat"].named_steps["enc"]
        unseen = [
            column for column, categories in zip(FEATURES, encoder.categories_)
            if pd.notna(values[column]) and values[column] not in categories
        ]
        return jsonify(
            prediction=predictions[saved["best"]], model=saved["best"],
            confidence=max(probabilities.values()), probabilities=probabilities,
            all_models=predictions, unseen_features=unseen,
        )

    return application


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port, debug=False)
