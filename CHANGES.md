# Cleanup record

- Replaced 52 incremental notebook cells with an ordered upload/setup/check/
  train-or-load/results/demo/download workflow. No patching of source files.
- Kept the 257-row CSV unchanged, including source links and review notes.
- Removed synthetic generation, old synthetic caches/results, backup source
  folders, the duplicate pending CSV and bytecode caches from the active project.
- Separated preprocessing/data validation, model definitions, training,
  evaluation and Flask inference. Deduplicated the per-model plotting code.
- Retained feature definitions, label boundaries, seeds, candidate group-split
  search, grouped CV, two algorithms and hyperparameter combinations.
- Fixed the missing network field in validation messages, malformed JSON/type
  handling, held-out option filling and frontend error reporting.
- App model reuse now checks dataset bytes, training code and scikit-learn
  version. Model fitting is never triggered by a browser request.
- Run metadata and a complete split/CV manifest document what generated results.
- Added a separate majority baseline and machine-readable SHAP/LIME diagnostics.
- Preserved the uploaded real-data results in reference_results; fresh results
  are clearly separated. No score was invented or copied into a new run.

The source dataset remains a candidate collection. This cleanup does not
independently verify every bank condition or establish stronger model accuracy.
