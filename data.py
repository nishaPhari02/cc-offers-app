"""Shared schema, validation and labels. Never generates or edits source data."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "offers.csv"
FEATURES = ["Bank", "Card Tier", "Merchant", "Category", "Card Network"]
DISCOUNT_COL = "Discount Percentage"
TARGET = "Discount Tier"
GROUP_COL = "Split Group"
LABELS = ["Low", "Medium", "High"]
SEED = 42
PIPELINE_VERSION = "clean-grouped-v1"
FIELD_MAP = dict(zip(["bank", "tier", "merchant", "category", "network"], FEATURES))


def dataset_hash(path=DATA_PATH):
    return sha256(Path(path).read_bytes()).hexdigest()


def add_labels(frame):
    frame = frame.copy()
    frame[TARGET] = pd.cut(
        frame[DISCOUNT_COL],
        bins=[-np.inf, 10, 20, np.inf],
        labels=LABELS,
    ).astype(str)
    return frame


def load_data(path=DATA_PATH):
    """Validate candidates, normalize missing inputs, deduplicate within groups.

    Source/provenance columns are retained. Numeric discount and target are
    never part of FEATURES. Missing card eligibility is not guessed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing: {path}")
    frame = pd.read_csv(path)
    required = FEATURES + [DISCOUNT_COL, GROUP_COL, "Source URL"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset has no rows.")

    for column in FEATURES + [GROUP_COL, "Source URL"]:
        values = frame[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        ).replace("", np.nan)
        frame[column] = values.astype(object).where(values.notna(), np.nan)

    for column in [GROUP_COL, "Source URL"]:
        if frame[column].isna().any():
            raise ValueError(f"Every row needs a nonblank {column}.")
        if not frame[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain text.")
    for column in FEATURES:
        if not frame[column].map(
            lambda value: pd.isna(value) or isinstance(value, str)
        ).all():
            raise ValueError(f"{column} must contain text or a blank.")
    if frame[FEATURES].isna().all(axis=1).any():
        raise ValueError("At least one prediction input is required on every row.")

    frame[DISCOUNT_COL] = pd.to_numeric(frame[DISCOUNT_COL], errors="coerce")
    invalid = frame[DISCOUNT_COL].isna() | ~frame[DISCOUNT_COL].between(0, 100)
    if invalid.any():
        raise ValueError(f"Invalid discounts on CSV rows: {(frame.index[invalid] + 2).tolist()}")
    # Different promotions may legitimately share the same inputs and rate.
    before = len(frame)
    frame = frame.drop_duplicates(
        subset=FEATURES + [DISCOUNT_COL, GROUP_COL]
    ).reset_index(drop=True)
    frame = add_labels(frame)
    if set(frame[TARGET]) != set(LABELS):
        raise ValueError("Training data must include Low, Medium and High examples.")
    frame.attrs["input_rows"] = before
    frame.attrs["duplicates_removed"] = before - len(frame)
    return frame


def partition_data(frame):
    """Retain the supplied run's split logic and deterministic demo group."""
    from grouped_training import split_real_data, make_group_cv

    demo = frame.sample(1, random_state=7)
    demo_group = demo[GROUP_COL].iloc[0]
    development = frame.loc[frame[GROUP_COL] != demo_group].copy()
    if set(development[TARGET]) != set(LABELS):
        raise ValueError("Demo-group removal would remove an entire class.")
    X = development[FEATURES]
    y = development[TARGET].map({label: i for i, label in enumerate(LABELS)}).astype(int)
    groups = development[GROUP_COL]
    positions = split_real_data(X, y, groups, seed=SEED)
    partitions = {}
    for name, pos in zip(["training", "validation", "testing"], positions):
        partitions[name] = {
            "X": X.iloc[pos], "y": y.iloc[pos], "groups": groups.iloc[pos]
        }
    training = partitions["training"]
    folds = make_group_cv(training["X"], training["y"], training["groups"], SEED)
    manifest = frame.copy()
    manifest["Partition"] = "demo"
    for name, part in partitions.items():
        manifest.loc[part["X"].index, "Partition"] = name
    # Include ALL reserved demo-group rows in the manifest, not only the sample.
    return partitions, demo, manifest, folds
