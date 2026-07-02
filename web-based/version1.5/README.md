# PaveLIFE-Lite Web v1.5

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

## v1.5 Ranking tab

This version adds a Ranking tab after the IRI & M&R, LCA, and LCCA result tabs.
The ranking function compares the calculated design(s) against the reference case databases stored in:

```text
data/ranking_database/GHG_emissions_all.xlsx
data/ranking_database/Costs_all.xlsx
```

Ranking convention: lower life-cycle GHG emissions or lower life-cycle cost ranks better. The current calculated design is inserted into the reference database; therefore ranks are reported as `rank / (reference cases + current design)`. The app reports both all-database ranking and same-pavement-type ranking.
