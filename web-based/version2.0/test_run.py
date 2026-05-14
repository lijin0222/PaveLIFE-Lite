# test_run.py

import os

from pavelife_core.runner import run_pavelife_single_section


model_dir = os.path.join(os.getcwd(), "model_artifacts")

input_dict = {
    "shrp_id": "demo_001",
    "pavement_family": "asphalt",
    "pavement_type": "ACUB",
    "iri0": 0.95,
    "precipitation": 800.0,
    "temperature": 15.0,
    "freeze_index": 100.0,
    "aadt": 8000.0,
    "aadtt": 800.0,
    "kesal": 500.0,
    # Web-version inputs are in meters. The runner converts a model-input copy to inches.
    "surface_thickness": 0.15,
    "base_thickness": 0.25,
    "subbase_thickness": 0.20,
    "func_class": 1,
}

analysis_settings = {
    "analysis_years": 50,
    "analysis_period_years": 50,
    "real_discount_rate": 0.04,
    "thickness_unit_to_m": 1.0,
}

results = run_pavelife_single_section(
    input_dict=input_dict,
    analysis_settings=analysis_settings,
    model_dir=model_dir,
    output_dir=os.path.join(os.getcwd(), "outputs"),
    make_figures=False,
    save_excel_outputs=False,
)

print(results.keys())
print(results["rate_prediction"][["rate_log_pred", "rate_pred"]])
print(results["maintenance_summary"])
print(results["lca_summary"][["Total Emission"]])
print(results["lcca_summary"][["Total Cost"]])
