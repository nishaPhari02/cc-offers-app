# Verification record

- Dataset: 257 rows; bytes identical to the uploaded offers.csv.
- Dataset SHA-256: 8da90f72656c50c003e038e6e0e0b21fa1096c6d2c8cfbda71f77c9c188fdd29
- Original real-data result files: preserved byte for byte in reference_results/.
- Full grouped training, tuning, EDA, per-model evaluation, SHAP and LIME completed.
- Original model comparison scores reproduced to four decimal places.
- Selected model: Extra Trees. Selection uses validation macro F1 only.
- Training/validation/test/demo rows: 179 / 34 / 43 / 1.
- Eight unittest checks passed with no skips after training.
- SHAP additivity and LIME prediction-wrapper consistency checked.
- LIME Extra Trees local R-squared reproduced: 0.132840. This remains a weak fit.
- Notebook: 8 code cells, all syntactically checked, with no stale saved outputs.
- Notebook data-validation, cache-reuse and regression-test steps executed locally.
- Live Flask + DOM interactions passed: metrics table, blank-input rejection,
  demo load/prediction, unseen input and clearing stale demo labels.
- Explanation charts visually inspected for legibility.

## Verification limits

A full Chromium visual run could not be completed because the browser binary
download failed. Page behaviour was checked with a DOM runner against the live
server; this does not establish pixel layout or mobile rendering. Colab-specific
file upload, iframe display and download require the user's Colab session and
were not executed locally. Their cells follow the previously working Colab APIs.

These checks do not independently verify bank terms, prove model reliability,
or create an untouched final test set. Existing dataset limitations remain.
