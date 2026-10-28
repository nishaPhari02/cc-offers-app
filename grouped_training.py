
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold
)


def split_real_data(X, y, groups, seed=42):
    """Keep related offers together in every partition."""
    classes = set(y.unique())
    overall = y.value_counts(normalize=True).reindex(
        sorted(classes), fill_value=0
    )
    best = None

    splitter = GroupShuffleSplit(
        n_splits=200,
        test_size=0.15,
        random_state=seed
    )

    for attempt, (dev_pos, test_pos) in enumerate(
        splitter.split(X, y, groups)
    ):
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=0.15 / 0.85,
            random_state=seed + attempt
        )

        train_rel, val_rel = next(inner.split(
            X.iloc[dev_pos],
            y.iloc[dev_pos],
            groups.iloc[dev_pos]
        ))

        train_pos = dev_pos[train_rel]
        val_pos = dev_pos[val_rel]
        parts = [train_pos, val_pos, test_pos]

        if any(
            set(y.iloc[p].unique()) != classes
            for p in parts
        ):
            continue

        train_group_counts = (
            groups.iloc[train_pos]
            .groupby(y.iloc[train_pos])
            .nunique()
        )

        if train_group_counts.min() < 2:
            continue

        balance_error = sum(
            (
                y.iloc[p].value_counts(normalize=True)
                .reindex(overall.index, fill_value=0)
                - overall
            ).abs().sum()
            for p in parts
        )

        size_error = sum(
            abs(len(p) / len(X) - fraction)
            for p, fraction in zip(
                parts, [0.70, 0.15, 0.15]
            )
        )

        score = balance_error + size_error

        if best is None or score < best[0]:
            best = (score, parts)

    if best is None:
        raise ValueError(
            "Cannot create suitable grouped splits. "
            "Review group sizes and class counts."
        )

    train_pos, val_pos, test_pos = best[1]

    group_sets = [
        set(groups.iloc[p])
        for p in [train_pos, val_pos, test_pos]
    ]

    assert not group_sets[0] & group_sets[1]
    assert not group_sets[0] & group_sets[2]
    assert not group_sets[1] & group_sets[2]

    return train_pos, val_pos, test_pos


def make_group_cv(X, y, groups, seed=42):
    """Use training-only groups during tuning."""
    classes = set(y.unique())
    class_groups = groups.groupby(y).nunique()
    maximum = min(3, int(class_groups.min()))

    for folds in range(maximum, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed
        )

        splits = list(splitter.split(X, y, groups))

        valid = all(
            set(y.iloc[a].unique()) == classes
            and set(y.iloc[b].unique()) == classes
            and not set(groups.iloc[a]) & set(groups.iloc[b])
            for a, b in splits
        )

        if valid:
            return splits

    raise ValueError(
        "Insufficient independent groups for tuning."
    )
