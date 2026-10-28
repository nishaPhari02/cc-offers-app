# Sri Lankan credit card offer discount tiers

Predict **Low**, **Medium** or **High** from Bank, Card Tier, Merchant,
Category and Card Network. The discount percentage creates the target and
is never supplied to a model as an input.

## Start in Google Colab

1. Open `Credit_Card_Offers_Colab.ipynb` in a fresh Colab runtime.
2. Run its cells from top to bottom. Upload the cleaned project ZIP when asked.
3. The notebook installs requirements, checks data, reuses a compatible model
   or trains one, shows results, and opens the app inside Colab.
4. Download your updated project using the last cell before leaving Colab.

Do not run the old 52-cell patch notebook on top of this project. The cleaned
notebook calls the saved Python modules instead of rewriting their source.
Python 3.12 or 3.13 is recommended with the pinned dependencies. Installation
must finish before training. If Colab asks to restart, restart and rerun setup.

## What is included

| File or folder | Purpose |
|---|---|
| `offers.csv` | Original 257-row source-linked candidate dataset, preserved byte for byte |
| `data.py` | Validation, target labels, demo reservation and grouped partitions |
| `grouped_training.py` | Group-aware split and cross-validation helpers |
| `models.py` | Random Forest, Extra Trees and the search grid |
| `train.py` | Explicit training, model selection, evaluation and explanation workflow |
| `evaluation.py` | Shared metrics, confusion matrices and permutation importance |
| `eda.py` | Data quality summaries and exploratory plots |
| `shap_explanations.py` | Global and individual SHAP outputs for both models |
| `lime_explanations.py` | Individual LIME explanations and fit diagnostics |
| `app.py`, `index.html` | Flask inference API and browser form |
| `models.joblib`, `model_metadata.json` | Current generated models and their provenance |
| `results/` | Outputs regenerated using the cleaned project |
| `reference_results/` | Your uploaded real-data run, unchanged, for comparison |
| `tests/` | Data, grouping, API, explanation and stale-model checks |
| `requirements.txt` | Pinned versions matching the supplied experiment |

Old patch cells, backup source copies, duplicate pending CSV, unused model
imports, synthetic generator, synthetic results/models and bytecode caches
are excluded. Original uploads remain available separately.

## Local commands

Run from the extracted `cc_offers_app` folder:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python app.py
```

Open `http://127.0.0.1:5000`. Stop the server with Ctrl+C.
The development server is sufficient for a local assignment demo.

To explicitly replace the current generated run after reviewing changes:

```bash
python train.py --overwrite
```

To create a run when no model exists:

```bash
python train.py
```

Standalone raw-data EDA:

```bash
python eda.py
```

Training regenerates EDA too. `train.py --data path/to/offers.csv --output
path/to/run` can write an isolated experiment. App defaults use the dataset
and model beside the Python files; use `create_app(data_path=..., model_dir=...)`
for programmatic testing of another run.

## Dataset and labels

The input is a **collection draft**, not a guarantee that all offer terms
have been independently confirmed. Keep `Evidence`, `Review Notes`, dates,
conditions and source links. Unknown card tiers/networks remain missing.
Do not turn a missing network into Visa or claim all tiers are eligible.

| Bank | Rows |
|---|---:|
| People's Bank | 116 |
| Nations Trust Bank | 72 |
| DFCC Bank | 41 |
| Seylan Bank | 16 |
| HNB | 11 |
| NDB | 1 |
| Total | 257 |

The CSV retains the original bank spellings. Classes are Low: 24, Medium: 139,
High: 94. Low is <=10%; Medium is >10% and <=20%; High is >20%.
Only the supplied fixed-rate candidate rows are included. No additional
random or invented offers are created.

Required columns: five model features above, `Discount Percentage`,
`Split Group` and `Source URL`. Percentages must be numeric and within 0–100.
Training rejects missing groups/sources or missing classes. It removes repeated
input/rate rows only within the same group; it does not rewrite `offers.csv`.

## Method retained from the supplied run

- One sample is drawn with seed 7; its entire group is reserved for the demo.
- The remaining rows are split by group, approximately 70/15/15. Candidate
  splits are scored only for class/size balance, never model performance.
- The supplied dataset gives 179 training, 34 validation, 43 test and 1 demo
  row. Generated group IDs are suggestions and still need campaign review.
- Missing categorical inputs are imputed as `Unknown` inside each fitted
  pipeline, then ordinal encoded. Unseen values use code -1.
- Both models use 300 trees and seed 42. The same eight combinations tune
  depth `[None, 10]`, minimum leaf size `[1, 3]`, and features per split
  `[sqrt, 1.0]`. Up to three stratified group folds use training data only.
- Choose the model using **unrounded validation macro F1**. Test results are
  calculated after selection. A most-frequent-class baseline is reported
  separately and never selected for serving.
- Do not switch to a model because its test score looks better.

`results/split_manifest.csv` includes every retained dataset row, including
all demo-group rows. `results/tuning_folds.csv` records CV membership.
Preprocessing is fitted inside the training folds, not on the full dataset.

## Reading results

- `model_comparison.csv`: validation macro F1 plus separate test metrics.
- `random_forest/`, `extra_trees/`: per-model report, confusion matrix and
  validation permutation importance. Negative importance is not an error;
  interpret it with its variation and small sample size.
- `shap/`: mean absolute SHAP across validation samples/classes and one
  predicted-class explanation. Baseline plus contributions is checked against
  the model probability.
- `lime/`: the same validation example, class weights, HTML/PNG and fit
  diagnostics. A close estimate for one row does not establish a good local
  fit. Report local R-squared and the prediction gap honestly.
- `model_metadata.json` / `results/run_metadata.json`: dataset hash, code hash,
  versions, settings, selected model and partition counts.

Saved explanations describe a validation example, not the current browser
input. They do not establish causal effects. LIME can perturb features into
unrealistic merchant/category combinations.

The uploaded run's results are in `reference_results/`; current outputs are
in `results/`. They are different runs, and library versions can affect scores.
Use one run consistently for your report. Both runs use an already-inspected
development test set; cleanup does not create new untouched evaluation data.

## Model reuse and API

The app never silently retrains on a request. If data, training code or the
scikit-learn version changed, it returns a clear error requiring explicit
training. Failed training stages do not replace an existing successful run.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Form |
| `/health` | GET | Ready status and dataset row count |
| `/options` | GET | Form choices |
| `/metrics` | GET | Model comparison and dataset context |
| `/holdout` | GET | Reserved demo sample |
| `/predict` | POST | Predict from `bank`, `tier`, `merchant`, `category`, `network` |

Blank inputs are allowed, but at least one must be supplied. Invalid JSON,
lists, non-text inputs and unexpected fields return 400. Unseen categories
are supported and identified in the response. Probabilities are estimates,
not guarantees or calibrated confidence levels.

## Remaining assignment work

This cleanup completes neither the written report nor data verification.
Confirm Extra Trees qualifies as not taught; review eligibility, campaigns,
class balance and sources; interpret the metrics, zero Low-class performance
and weak LIME fits; discuss bias/ethics and limitations; and record the 3–5
minute demo. If improving the model after inspecting test results, use a
fresh untouched evaluation set for new final performance claims.
