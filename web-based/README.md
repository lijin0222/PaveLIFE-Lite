# PaveLIFE-Lite Web

This folder contains a Streamlit web version of `PaveLIFE-Lite_v1.0.ipynb`.

## Required model artifacts

Following files in `model_artifacts/`:

- `model_xgb_asphalt.joblib`
- `feature_columns_asphalt.joblib`
- `onehot_encoder_asphalt.joblib`
- `scaler_asphalt.joblib`
- `model_xgb_concrete.joblib`
- `feature_columns_concrete.joblib`
- `onehot_encoder_concrete.joblib`
- `scaler_concrete.joblib`

## Run in your environment

```bash
conda activate your_env_name
cd PaveLIFE_Streamlit_Web
streamlit run app.py
```

## Optional command-line smoke test

```bash
conda activate your_env_name
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
