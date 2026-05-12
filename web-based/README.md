# PaveLIFE-Lite Web

This folder contains a Streamlit web version of `PaveLIFE-Lite_v1.0.ipynb`.

## Required model artifacts

Place the following files in `model_artifacts/`:

- `model_xgb_asphalt.joblib`
- `feature_columns_asphalt.joblib`
- `onehot_encoder_asphalt.joblib`
- `scaler_asphalt.joblib`
- `model_xgb_concrete.joblib`
- `feature_columns_concrete.joblib`
- `onehot_encoder_concrete.joblib`
- `scaler_concrete.joblib`

If you only run asphalt sections, the asphalt artifacts are enough. If you only run concrete sections, the concrete artifacts are enough.

## Run in your existing MI environment

```bash
conda activate MI
cd PaveLIFE_Streamlit_Web
streamlit run app.py
```

## Optional command-line smoke test

```bash
conda activate MI
cd PaveLIFE_Streamlit_Web
python test_run.py
```

## Main files

- `app.py`: Streamlit web interface.
- `pavelife_core/legacy.py`: compatibility calculation layer generated from the original notebook.
- `pavelife_core/runner.py`: clean single-section workflow called by the web app.
- `pavelife_core/schemas.py`: input normalization and validation.
- `pavelife_core/model_loader.py`: model-artifact loading.
- `pavelife_core/outputs.py`: in-memory Excel export.

## Design note

The current version keeps the original calculation logic in `legacy.py` to preserve consistency with the notebook. The web app uses `runner.py` as the stable interface. In later versions, the functions inside `legacy.py` can be further split into smaller fully independent modules.
