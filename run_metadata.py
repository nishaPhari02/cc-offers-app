"""Small, readable provenance records for model reuse and result interpretation."""

from hashlib import sha256
from importlib.metadata import version
import platform

from data import PROJECT_DIR, PIPELINE_VERSION

PACKAGES = ["scikit-learn", "numpy", "pandas", "joblib", "flask", "matplotlib", "shap", "lime"]


def package_versions():
    return {"python": platform.python_version(), **{name: version(name) for name in PACKAGES}}


def training_code_hash():
    digest = sha256()
    for name in [
        "data.py", "models.py", "grouped_training.py", "evaluation.py",
        "eda.py", "shap_explanations.py", "lime_explanations.py", "train.py",
    ]:
        digest.update(name.encode())
        digest.update((PROJECT_DIR / name).read_bytes())
    return digest.hexdigest()


def cache_is_current(metadata, data_sha256):
    return (
        metadata.get("pipeline_version") == PIPELINE_VERSION
        and metadata.get("dataset_sha256") == data_sha256
        and metadata.get("training_code_sha256") == training_code_hash()
        and metadata.get("versions", {}).get("scikit-learn") == version("scikit-learn")
    )
