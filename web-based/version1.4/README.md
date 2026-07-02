# PaveLIFE-Lite Web v2.0

This version adds two analysis levels:

1. **Single pavement design**: evaluate one pavement design.
2. **Design comparison**: compare up to 10 pavement designs under the same IRI0, functional class, climate, traffic, and analysis assumptions.

Both analysis levels support **Basic mode** and **Expert mode**.

## Run

```bash
conda activate MI
cd path/to/PaveLIFE_Streamlit_Web
python -m streamlit run app.py
```

## Model artifacts

Place the pretrained model files in `model_artifacts/` or specify their folder in the sidebar:

- `model_xgb_asphalt.joblib`
- `feature_columns_asphalt.joblib`
- `onehot_encoder_asphalt.joblib`
- `scaler_asphalt.joblib`
- `model_xgb_concrete.joblib`
- `feature_columns_concrete.joblib`
- `onehot_encoder_concrete.joblib`
- `scaler_concrete.joblib`

## Design comparison mode

In comparison mode, each design can independently set:

- Pavement family
- Pavement type
- Surface thickness, m
- Base thickness, m
- Subbase thickness, m

The following are shared across all compared designs:

- Initial IRI
- Functional class
- Precipitation
- Temperature
- Freeze index
- AADT
- AADTT
- KESAL
- Analysis period
- Discount rate
- Expert-mode LCA/LCCA assumptions
