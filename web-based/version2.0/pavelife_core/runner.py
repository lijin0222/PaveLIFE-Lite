# pavelife_core/runner.py

import os

import pandas as pd

from pavelife_core import legacy
from pavelife_core.config import default_analysis_settings, func_class_lookup
from pavelife_core.schemas import validate_section_input
from pavelife_core.rate_prediction import build_single_section_dataframe, predict_rate_for_single_section
from pavelife_core.iri import process_pavement_threshold_by_predicted_rate
from pavelife_core.lca import run_lca_single_section
from pavelife_core.lcca import run_lcca_single_section
from pavelife_core.outputs import (
    build_phase_contribution_df,
    build_layer_stage_table,
    build_parameter_documentation_df,
    build_rate_metadata_df,
    build_pavement_type_config_df,
)
from pavelife_core.model_loader import load_pretrained_rate_artifacts


def sync_derived_parameters():
    """Refresh derived globals after expert-mode parameter updates."""
    legacy.pavement_area = legacy.lane_length * legacy.lane_width

    legacy.gasoline_price_per_liter = legacy.gasoline_price_per_gallon / legacy.liter_per_gallon
    legacy.diesel_price_per_liter = legacy.diesel_price_per_gallon / legacy.liter_per_gallon
    legacy.passenger_vehicle_delay_value_per_hour = (
        legacy.passenger_vehicle_occupancy * legacy.passenger_time_value_per_person_hour
    )

    legacy.lca_material_prod_factor_lookup = {
        "asphalt": legacy.asphalt_prod_factor,
        "concrete": legacy.concrete_prod_factor,
        "aggregate": legacy.aggregate_prod_factor,
        "ctb": legacy.ctb_prod_factor,
        "atb": legacy.atb_prod_factor,
        "rebar": legacy.rebar_prod_factor,
    }

    legacy.lcca_material_cost_lookup = {
        "asphalt": legacy.asphalt_material_cost_per_ton,
        "aggregate": legacy.aggregate_material_cost_per_ton,
        "ctb": legacy.ctb_material_cost_per_ton,
        "atb": legacy.atb_material_cost_per_ton,
        "concrete": legacy.concrete_material_cost_per_ton,
        "rebar": legacy.rebar_material_cost_per_ton,
    }

    legacy.transport_distance_lookup = {
        "asphalt": legacy.asphalt_transport_distance,
        "concrete": legacy.concrete_transport_distance,
        "base": legacy.base_transport_distance,
        "subbase": legacy.subbase_transport_distance,
        "aggregate": legacy.aggregate_transport_distance,
    }

    legacy.lcca_construction_cost_lookup = {
        "asphalt": legacy.asphalt_construction_cost_per_ton,
        "base": legacy.base_construction_cost_per_ton,
        "subbase": legacy.subbase_construction_cost_per_ton,
        "concrete": legacy.concrete_paving_cost_per_ton,
    }

    legacy.lca_demolition_fuel_lookup = {
        "granular": legacy.demolition_granular_l_per_ton,
        "ctb": legacy.demolition_ctb_l_per_ton,
        "atb": legacy.demolition_atb_l_per_ton,
        "stabilized": legacy.demolition_stabilized_l_per_ton,
        "concrete": legacy.demolition_concrete_l_per_ton,
    }


def apply_analysis_settings(analysis_settings=None, output_dir=None):
    """Apply user-selected analysis settings to the calculation layer."""
    settings = default_analysis_settings.copy()
    if analysis_settings is not None:
        settings.update(analysis_settings)

    for key, value in settings.items():
        if hasattr(legacy, key):
            setattr(legacy, key, value)

    if "analysis_years" in settings:
        legacy.analysis_years = int(settings["analysis_years"])

    if "analysis_period_years" in settings:
        legacy.analysis_period_years = int(settings["analysis_period_years"])

    if output_dir is not None:
        legacy.base_output_dir = os.path.abspath(output_dir)
        legacy.iri_output_dir = os.path.join(legacy.base_output_dir, "life_cycle_iri")
        legacy.lca_output_dir = os.path.join(legacy.base_output_dir, "lca")
        legacy.lcca_output_dir = os.path.join(legacy.base_output_dir, "lcca")
        legacy.figure_output_dir = os.path.join(legacy.base_output_dir, "figures")
        for folder in [
            legacy.base_output_dir,
            legacy.iri_output_dir,
            legacy.lca_output_dir,
            legacy.lcca_output_dir,
            legacy.figure_output_dir,
        ]:
            os.makedirs(folder, exist_ok=True)

    sync_derived_parameters()
    return settings


def get_model_tuple(pavement_family, model_artifacts=None, model_dir=None):
    """Return model artifacts for one pavement family."""
    if model_artifacts is not None and pavement_family in model_artifacts:
        return model_artifacts[pavement_family]

    return load_pretrained_rate_artifacts(pavement_family=pavement_family, model_dir=model_dir)


def build_rate_model_input_from_meter_input(df_input):
    """
    Create a temporary model-input DataFrame with layer thicknesses in inches.

    The Streamlit UI asks users to enter thicknesses in meters because LCA/LCCA
    material quantities are calculated in SI units. The pretrained deterioration
    models, however, were trained with thickness-related variables in inches.
    This helper only converts the copy used for rate prediction and leaves the
    original df_input unchanged for LCA/LCCA.
    """
    df_model_input = df_input.copy()
    meter_to_inch = 1.0 / 0.0254

    thickness_columns = [
        "Surface thickness",
        "Base thickness",
        "Subbase thickness",
        "Asphalt thickness",
        "Concrete thickness",
        "PCC thickness",
        "Slab thickness",
        "Concrete slab thickness",
    ]

    for col in thickness_columns:
        if col in df_model_input.columns:
            df_model_input[col] = pd.to_numeric(df_model_input[col], errors="coerce") * meter_to_inch

    return df_model_input


def run_pavelife_single_section(
    input_dict,
    model_artifacts=None,
    analysis_settings=None,
    model_dir=None,
    output_dir=None,
    make_figures=False,
    save_excel_outputs=False,
):
    """Run the full PaveLIFE workflow for one user-defined section."""
    settings = apply_analysis_settings(analysis_settings=analysis_settings, output_dir=output_dir)

    legacy.make_figures = bool(make_figures)
    legacy.save_excel_outputs = bool(save_excel_outputs)
    legacy.show_figures = False

    section_input = validate_section_input(input_dict)
    df_input = build_single_section_dataframe(section_input)

    pavement_family = str(df_input["Pavement family"].iloc[0]).lower().strip()
    pavement_type = str(df_input["Pavement type"].iloc[0]).upper().strip()

    (
        rate_model,
        rate_scaler,
        rate_encoder,
        rate_feature_columns,
        rate_metadata,
    ) = get_model_tuple(
        pavement_family=pavement_family,
        model_artifacts=model_artifacts,
        model_dir=model_dir,
    )

    df_model_input = build_rate_model_input_from_meter_input(df_input)

    df_rate_pred_model_units = predict_rate_for_single_section(
        df_single=df_model_input,
        rate_model=rate_model,
        rate_scaler=rate_scaler,
        rate_encoder=rate_encoder,
        rate_metadata=rate_metadata,
    )

    # Keep all user-facing and LCA/LCCA thickness columns in meters, while
    # retaining the predicted deterioration-rate outputs from the model-input copy.
    df_rate_pred = df_input.copy()
    df_rate_pred["rate_log_pred"] = df_rate_pred_model_units["rate_log_pred"].values
    df_rate_pred["rate_pred"] = df_rate_pred_model_units["rate_pred"].values
    df_rate_pred["thickness_input_unit"] = "m"
    df_rate_pred["thickness_model_unit"] = "inch"

    row = df_rate_pred.iloc[0]

    df_yearly, maintenance_summary_df, maintenance_years = process_pavement_threshold_by_predicted_rate(
        row=row,
        analysis_years_value=int(settings["analysis_years"]),
    )

    lca_summary, lca_maintenance_events, lca_eol_details = run_lca_single_section(
        row=row,
        df_yearly=df_yearly,
        maintenance_years=maintenance_years,
    )

    lcca_summary, lcca_maintenance_events, lcca_eol_details = run_lcca_single_section(
        row=row,
        df_yearly=df_yearly,
        maintenance_years=maintenance_years,
    )

    lca_phase_contribution = build_phase_contribution_df(
        summary_df=lca_summary,
        total_column="Total Emission",
        value_unit="kg CO2e",
    )

    lcca_phase_contribution = build_phase_contribution_df(
        summary_df=lcca_summary,
        total_column="Total Cost",
        value_unit=f"{int(settings['analysis_base_year'])} USD PV",
    )

    lca_layer_stage = build_layer_stage_table(lca_summary, "kg CO2e")
    lcca_layer_stage = build_layer_stage_table(lcca_summary, f"{int(settings['analysis_base_year'])} USD PV")
    parameter_documentation_df = build_parameter_documentation_df()
    rate_metadata_df = build_rate_metadata_df(rate_metadata)
    func_class_mapping_df = (
        pd.DataFrame.from_dict(func_class_lookup, orient="index")
        .reset_index()
        .rename(columns={"index": "FUNC_CLASS"})
    )
    pavement_type_config_df = build_pavement_type_config_df()

    shrp_id = str(df_input["SHRP_ID"].iloc[0])
    safe_shrp_id = legacy.safe_file_name(shrp_id)

    output_file = os.path.join(
        legacy.base_output_dir,
        f"PaveLIFE_LCA_LCCA_{pavement_family}_{pavement_type}_{safe_shrp_id}_single_section.xlsx",
    )

    iri_file = os.path.join(
        legacy.iri_output_dir,
        f"PaveLIFE_IRI_{pavement_family}_{pavement_type}_{safe_shrp_id}_single_section.xlsx",
    )

    results = {
        "input_parameters": df_input,
        "rate_prediction": df_rate_pred,
        "yearly_iri": df_yearly,
        "maintenance_summary": maintenance_summary_df,
        "lca_summary": lca_summary,
        "lca_phase_contribution": lca_phase_contribution,
        "lca_layer_stage": lca_layer_stage,
        "lca_maintenance_events": lca_maintenance_events,
        "lca_eol_details": lca_eol_details,
        "lcca_summary": lcca_summary,
        "lcca_phase_contribution": lcca_phase_contribution,
        "lcca_layer_stage": lcca_layer_stage,
        "lcca_maintenance_events": lcca_maintenance_events,
        "lcca_eol_details": lcca_eol_details,
        "parameter_documentation": parameter_documentation_df,
        "rate_model_metadata": rate_metadata_df,
        "func_class_mapping": func_class_mapping_df,
        "pavement_type_config": pavement_type_config_df,
        "output_file": output_file,
        "iri_file": iri_file,
        "figure_paths": {},
    }

    if make_figures:
        results["figure_paths"] = legacy.create_all_visualizations(results)

    if save_excel_outputs:
        legacy.save_combined_outputs(
            output_file=output_file,
            df_input=df_input,
            df_rate_pred=df_rate_pred,
            df_yearly=df_yearly,
            maintenance_summary_df=maintenance_summary_df,
            lca_summary=lca_summary,
            lca_phase_contribution=lca_phase_contribution,
            lca_layer_stage=lca_layer_stage,
            lca_maintenance_events=lca_maintenance_events,
            lca_eol_details=lca_eol_details,
            lcca_summary=lcca_summary,
            lcca_phase_contribution=lcca_phase_contribution,
            lcca_layer_stage=lcca_layer_stage,
            lcca_maintenance_events=lcca_maintenance_events,
            lcca_eol_details=lcca_eol_details,
            parameter_documentation_df=parameter_documentation_df,
            rate_metadata_df=rate_metadata_df,
            func_class_mapping_df=func_class_mapping_df,
            pavement_type_config_df=pavement_type_config_df,
        )

        with pd.ExcelWriter(iri_file, engine="openpyxl") as writer:
            df_yearly.to_excel(writer, sheet_name="yearly_prediction", index=False)
            maintenance_summary_df.to_excel(writer, sheet_name="maintenance_summary", index=False)

    return results
