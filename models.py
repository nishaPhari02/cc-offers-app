"""The two compared algorithms and the fixed, training-only search space."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from data import FEATURES, SEED

PARAMETER_GRID = {
    "clf__max_depth": [None, 10],
    "clf__min_samples_leaf": [1, 3],
    "clf__max_features": ["sqrt", 1.0],
}


def preprocessing():
    # Retain ordinal encoding from the supplied experiment. Discuss its
    # arbitrary order for nominal categories as a modelling limitation.
    return ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("enc", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )),
        ]), FEATURES),
    ])


def build_models():
    # Parallelize GridSearchCV; each individual forest uses one worker.
    return {
        name: Pipeline([
            ("prep", preprocessing()),
            ("clf", classifier(n_estimators=300, random_state=SEED, n_jobs=1)),
        ])
        for name, classifier in [
            ("Random Forest", RandomForestClassifier),
            ("Extra Trees", ExtraTreesClassifier),
        ]
    }
