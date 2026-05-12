# app.py

import os

import pandas as pd
import streamlit as st

from pavelife_core.config import default_analysis_settings, func_class_lookup
from pavelife_core.model_loader import load_pretrained_rate_artifacts
from pavelife_core.outputs import build_excel_bytes
from pavelife_core.runner import run_pavelife_single_section


st.set_page_config(
    page_title="PaveLIFE-Lite",
    page_icon="🛣️",
    layout="wide",
)


@st.cache_resource
def get_model_artifact(pavement_family, model_dir):
    """Load only the model artifact required by the selected pavement family."""
    return load_pretrained_rate_artifacts(
        pavement_family=pavement_family,
        model_dir=model_dir,
    )


def get_float_from_df(df, column_name, default_value=0.0):
    if isinstance(df, pd.DataFrame) and column_name in df.columns and len(df) > 0:
        return float(df[column_name].iloc[0])
    return float(default_value)


def get_int_from_df(df, column_name, default_value=0):
    if isinstance(df, pd.DataFrame) and column_name in df.columns and len(df) > 0:
        return int(df[column_name].iloc[0])
    return int(default_value)


def render_contribution_chart(contribution_df, label):
    if not isinstance(contribution_df, pd.DataFrame):
        return

    required_cols = ["Phase label", "Contribution rate (%)"]
    if all(col in contribution_df.columns for col in required_cols):
        chart_df = contribution_df[required_cols].copy()
        chart_df = chart_df.set_index("Phase label")
        st.bar_chart(chart_df, use_container_width=True)
    else:
        st.caption(f"{label} contribution chart skipped because expected columns were not found.")


def render_iri_chart(df_yearly):
    if not isinstance(df_yearly, pd.DataFrame):
        return

    iri_cols = [col for col in ["IRI_pred_before_maintenance", "IRI_pred"] if col in df_yearly.columns]
    if "Pavement_Age" in df_yearly.columns and iri_cols:
        chart_df = df_yearly[["Pavement_Age"] + iri_cols].copy()
        chart_df = chart_df.set_index("Pavement_Age")
        st.line_chart(chart_df, use_container_width=True)
    else:
        st.caption("IRI chart skipped because expected columns were not found.")


def build_sidebar_inputs():
    with st.sidebar:
        st.title("PaveLIFE-Lite")
        st.caption("Single-section pavement LCA-LCCA web tool")

        mode = st.radio(
            "Input mode",
            options=["Basic mode", "Expert mode"],
            index=0,
            help="Basic mode exposes only core user inputs. Expert mode exposes selected LCA/LCCA assumptions.",
        )

        st.divider()
        st.header("1. Pavement section")

        shrp_id = st.text_input(
            "Section ID",
            value="demo_001",
            help="User-defined section or project ID.",
        )

        pavement_family = st.selectbox(
            "Pavement family",
            options=["asphalt", "concrete"],
            index=0,
        )

        if pavement_family == "asphalt":
            pavement_type_options = ["ACUB", "ACATB", "ACCTB", "FDA", "IP"]
        else:
            pavement_type_options = ["JPCP", "JRCP", "CRCP"]

        pavement_type = st.selectbox(
            "Pavement type",
            options=pavement_type_options,
            index=0,
        )

        iri0 = st.number_input(
            "Initial IRI, m/km",
            min_value=0.01,
            value=0.95,
            step=0.05,
            help="Initial International Roughness Index.",
        )

        func_class = st.selectbox(
            "Functional class",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            index=0,
        )

        func_info = func_class_lookup[int(func_class)]
        st.info(
            f"**{func_info['FUNC_CLASS_EXP']}**\n\n"
            f"IRI threshold: {func_info['IRI_threshold']} m/km\n\n"
            f"Car speed limit: {func_info['car_speed_limit_kph']} km/h\n\n"
            f"Truck speed limit: {func_info['truck_speed_limit_kph']} km/h"
        )

        st.header("2. Climate and traffic")

        precipitation = st.number_input("Precipitation", min_value=0.0, value=800.0, step=10.0)
        temperature = st.number_input("Temperature", value=15.0, step=1.0)
        freeze_index = st.number_input("Freeze index", min_value=0.0, value=100.0, step=10.0)
        aadt = st.number_input("AADT, vehicles/day", min_value=0.0, value=8000.0, step=100.0)
        aadtt = st.number_input("AADTT, trucks/day", min_value=0.0, value=800.0, step=50.0)
        kesal = st.number_input("KESAL", min_value=0.0, value=500.0, step=10.0)

        st.header("3. Layer thickness")

        surface_thickness = st.number_input(
            "Surface thickness",
            min_value=0.0,
            value=6.0,
            step=0.5,
            help="Asphalt layer thickness for asphalt pavements; concrete slab thickness for concrete pavements.",
        )
        base_thickness = st.number_input("Base thickness", min_value=0.0, value=10.0, step=0.5)
        subbase_thickness = st.number_input("Subbase thickness", min_value=0.0, value=8.0, step=0.5)

        st.header("4. Analysis settings")

        model_dir_default = os.path.join(os.getcwd(), "model_artifacts")
        model_dir = st.text_input(
            "Model artifact folder",
            value=model_dir_default,
            help="Folder containing the pretrained XGBoost, feature_columns, encoder, and scaler joblib files.",
        )

        analysis_years = st.number_input(
            "Analysis period, years",
            min_value=1,
            max_value=100,
            value=int(default_analysis_settings["analysis_years"]),
            step=1,
        )

        real_discount_rate = st.number_input(
            "Real discount rate",
            min_value=0.0,
            max_value=0.20,
            value=float(default_analysis_settings["real_discount_rate"]),
            step=0.005,
            format="%.3f",
        )

        thickness_unit = st.selectbox(
            "Thickness conversion",
            options=[
                "Original PaveLIFE convention: value × 0.025 m",
                "Exact inch-to-meter: value × 0.0254 m",
                "Already in meter: value × 1.0 m",
            ],
            index=0,
        )

        if thickness_unit.startswith("Original"):
            thickness_unit_to_m = 0.025
        elif thickness_unit.startswith("Exact"):
            thickness_unit_to_m = 0.0254
        else:
            thickness_unit_to_m = 1.0

        analysis_settings = default_analysis_settings.copy()
        analysis_settings["analysis_years"] = int(analysis_years)
        analysis_settings["analysis_period_years"] = int(analysis_years)
        analysis_settings["real_discount_rate"] = float(real_discount_rate)
        analysis_settings["thickness_unit_to_m"] = float(thickness_unit_to_m)

        if mode == "Expert mode":
            with st.expander("Expert: geometry and traffic", expanded=False):
                analysis_settings["analysis_base_year"] = int(
                    st.number_input(
                        "Analysis base year",
                        min_value=1900,
                        max_value=2200,
                        value=int(default_analysis_settings["analysis_base_year"]),
                        step=1,
                    )
                )
                analysis_settings["growth_rate"] = float(
                    st.number_input(
                        "Annual traffic growth rate",
                        min_value=0.0,
                        max_value=0.20,
                        value=float(default_analysis_settings["growth_rate"]),
                        step=0.005,
                        format="%.3f",
                    )
                )
                analysis_settings["lane_length"] = float(
                    st.number_input(
                        "Lane length, m",
                        min_value=1.0,
                        value=float(default_analysis_settings["lane_length"]),
                        step=100.0,
                    )
                )
                analysis_settings["lane_width"] = float(
                    st.number_input(
                        "Lane width, m",
                        min_value=1.0,
                        value=float(default_analysis_settings["lane_width"]),
                        step=0.1,
                    )
                )

            with st.expander("Expert: work-zone and maintenance", expanded=False):
                analysis_settings["work_zone_speed_kph"] = float(
                    st.number_input(
                        "Work-zone speed, km/h",
                        min_value=1.0,
                        value=float(default_analysis_settings["work_zone_speed_kph"]),
                        step=5.0,
                    )
                )
                analysis_settings["work_hours_per_day"] = float(
                    st.number_input(
                        "Work hours per day",
                        min_value=1.0,
                        value=float(default_analysis_settings["work_hours_per_day"]),
                        step=1.0,
                    )
                )
                analysis_settings["overlay_only_duration_days"] = float(
                    st.number_input(
                        "Overlay-only duration, days",
                        min_value=0.0,
                        value=float(default_analysis_settings["overlay_only_duration_days"]),
                        step=1.0,
                    )
                )
                analysis_settings["mill_and_overlay_duration_days"] = float(
                    st.number_input(
                        "Mill-and-overlay duration, days",
                        min_value=0.0,
                        value=float(default_analysis_settings["mill_and_overlay_duration_days"]),
                        step=1.0,
                    )
                )
                analysis_settings["cpr_diamond_grinding_duration_days"] = float(
                    st.number_input(
                        "CPR / diamond grinding duration, days",
                        min_value=0.0,
                        value=float(default_analysis_settings["cpr_diamond_grinding_duration_days"]),
                        step=1.0,
                    )
                )

            with st.expander("Expert: LCA emission factors", expanded=False):
                analysis_settings["asphalt_prod_factor"] = float(
                    st.number_input("Asphalt production EF, kg CO₂e/ton", value=float(default_analysis_settings["asphalt_prod_factor"]), step=1.0)
                )
                analysis_settings["concrete_prod_factor"] = float(
                    st.number_input("Concrete production EF, kg CO₂e/ton", value=float(default_analysis_settings["concrete_prod_factor"]), step=1.0)
                )
                analysis_settings["aggregate_prod_factor"] = float(
                    st.number_input("Aggregate production EF, kg CO₂e/ton", value=float(default_analysis_settings["aggregate_prod_factor"]), step=0.5)
                )
                analysis_settings["ctb_prod_factor"] = float(
                    st.number_input("CTB production EF, kg CO₂e/ton", value=float(default_analysis_settings["ctb_prod_factor"]), step=1.0)
                )
                analysis_settings["atb_prod_factor"] = float(
                    st.number_input("ATB production EF, kg CO₂e/ton", value=float(default_analysis_settings["atb_prod_factor"]), step=1.0)
                )
                analysis_settings["rebar_prod_factor"] = float(
                    st.number_input("Rebar production EF, kg CO₂e/ton", value=float(default_analysis_settings["rebar_prod_factor"]), step=50.0)
                )

            with st.expander("Expert: LCCA assumptions", expanded=False):
                analysis_settings["gasoline_price_per_gallon"] = float(
                    st.number_input("Gasoline price, USD/gal", value=float(default_analysis_settings["gasoline_price_per_gallon"]), step=0.1)
                )
                analysis_settings["diesel_price_per_gallon"] = float(
                    st.number_input("Diesel price, USD/gal", value=float(default_analysis_settings["diesel_price_per_gallon"]), step=0.1)
                )
                analysis_settings["asphalt_material_cost_per_ton"] = float(
                    st.number_input("Asphalt material cost, USD/ton", value=float(default_analysis_settings["asphalt_material_cost_per_ton"]), step=1.0)
                )
                analysis_settings["concrete_material_cost_per_ton"] = float(
                    st.number_input("Concrete material cost, USD/ton", value=float(default_analysis_settings["concrete_material_cost_per_ton"]), step=1.0)
                )
                analysis_settings["aggregate_material_cost_per_ton"] = float(
                    st.number_input("Aggregate material cost, USD/ton", value=float(default_analysis_settings["aggregate_material_cost_per_ton"]), step=1.0)
                )
                analysis_settings["reinforcement_ratio"] = float(
                    st.number_input("Reinforcement ratio for JRCP/CRCP", min_value=0.0, max_value=0.20, value=float(default_analysis_settings["reinforcement_ratio"]), step=0.001, format="%.4f")
                )

        run_button = st.button("Run PaveLIFE", type="primary", use_container_width=True)

    input_dict = {
        "shrp_id": shrp_id,
        "pavement_family": pavement_family,
        "pavement_type": pavement_type,
        "iri0": iri0,
        "precipitation": precipitation,
        "temperature": temperature,
        "freeze_index": freeze_index,
        "aadt": aadt,
        "aadtt": aadtt,
        "kesal": kesal,
        "surface_thickness": surface_thickness,
        "base_thickness": base_thickness,
        "subbase_thickness": subbase_thickness,
        "func_class": func_class,
    }

    return input_dict, analysis_settings, model_dir, run_button, mode


st.title("PaveLIFE-Lite Web")
st.markdown(
    "An interactive single-section pavement LCA-LCCA tool integrating "
    "pretrained IRI deterioration-rate prediction, threshold-triggered M&R simulation, "
    "life-cycle GHG accounting, and life-cycle cost analysis."
)

input_dict, analysis_settings, model_dir, run_button, mode = build_sidebar_inputs()

if "pavelife_results" not in st.session_state:
    st.session_state["pavelife_results"] = None

if not run_button and st.session_state["pavelife_results"] is None:
    st.info("Enter section-level parameters in the sidebar and click **Run PaveLIFE**.")
    st.stop()

if run_button:
    try:
        with st.spinner("Running PaveLIFE calculation..."):
            pavement_family = input_dict["pavement_family"]
            model_artifact = get_model_artifact(pavement_family, model_dir)
            model_artifacts = {pavement_family: model_artifact}

            results = run_pavelife_single_section(
                input_dict=input_dict,
                model_artifacts=model_artifacts,
                analysis_settings=analysis_settings,
                model_dir=model_dir,
                output_dir=os.path.join(os.getcwd(), "outputs"),
                make_figures=False,
                save_excel_outputs=False,
            )

        st.session_state["pavelife_results"] = results
        st.session_state["pavelife_input"] = input_dict
        st.session_state["pavelife_settings"] = analysis_settings
        st.success("PaveLIFE calculation completed.")

    except Exception as error:
        st.error("PaveLIFE calculation failed.")
        st.exception(error)
        st.stop()

results = st.session_state["pavelife_results"]
input_dict_current = st.session_state.get("pavelife_input", input_dict)

rate_prediction = results["rate_prediction"]
maintenance_summary = results["maintenance_summary"]
lca_summary = results["lca_summary"]
lcca_summary = results["lcca_summary"]

rate_pred = get_float_from_df(rate_prediction, "rate_pred", default_value=0.0)
rate_log_pred = get_float_from_df(rate_prediction, "rate_log_pred", default_value=0.0)
maintenance_count = get_int_from_df(maintenance_summary, "maintenance_count", default_value=0)
total_ghg = get_float_from_df(lca_summary, "Total Emission", default_value=0.0)
total_cost = get_float_from_df(lcca_summary, "Total Cost", default_value=0.0)

col_1, col_2, col_3, col_4 = st.columns(4)
col_1.metric("Predicted rate", f"{rate_pred:.6f}")
col_2.metric("Maintenance count", f"{maintenance_count}")
col_3.metric("Total GHG", f"{total_ghg:,.0f} kg CO₂e")
col_4.metric("Total cost PV", f"${total_cost:,.0f}")

st.caption(f"Input mode: {mode}. Predicted rate_log: {rate_log_pred:.6f}.")

tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs(
    ["IRI & M&R", "LCA", "LCCA", "Model and assumptions", "Download"]
)

with tab_1:
    st.subheader("IRI progression")
    render_iri_chart(results["yearly_iri"])

    st.subheader("Yearly IRI table")
    st.dataframe(results["yearly_iri"], use_container_width=True)

    st.subheader("Maintenance summary")
    st.dataframe(results["maintenance_summary"], use_container_width=True)

with tab_2:
    st.subheader("LCA contribution rate")
    render_contribution_chart(results["lca_phase_contribution"], "LCA")

    st.subheader("LCA summary")
    st.dataframe(results["lca_summary"], use_container_width=True)

    st.subheader("LCA phase contribution")
    st.dataframe(results["lca_phase_contribution"], use_container_width=True)

    st.subheader("LCA layer-stage table")
    st.dataframe(results["lca_layer_stage"], use_container_width=True)

    st.subheader("LCA maintenance events")
    st.dataframe(results["lca_maintenance_events"], use_container_width=True)

    st.subheader("LCA EOL details")
    st.dataframe(results["lca_eol_details"], use_container_width=True)

with tab_3:
    st.subheader("LCCA contribution rate")
    render_contribution_chart(results["lcca_phase_contribution"], "LCCA")

    st.subheader("LCCA summary")
    st.dataframe(results["lcca_summary"], use_container_width=True)

    st.subheader("LCCA phase contribution")
    st.dataframe(results["lcca_phase_contribution"], use_container_width=True)

    st.subheader("LCCA layer-stage table")
    st.dataframe(results["lcca_layer_stage"], use_container_width=True)

    st.subheader("LCCA maintenance events")
    st.dataframe(results["lcca_maintenance_events"], use_container_width=True)

    st.subheader("LCCA EOL details")
    st.dataframe(results["lcca_eol_details"], use_container_width=True)

with tab_4:
    st.subheader("Input parameters")
    st.dataframe(results["input_parameters"], use_container_width=True)

    st.subheader("Rate prediction")
    st.dataframe(results["rate_prediction"], use_container_width=True)

    st.subheader("Rate model metadata")
    st.dataframe(results["rate_model_metadata"], use_container_width=True)

    st.subheader("FUNC_CLASS mapping")
    st.dataframe(results["func_class_mapping"], use_container_width=True)

    st.subheader("Pavement type configuration")
    st.dataframe(results["pavement_type_config"], use_container_width=True)

    st.subheader("Parameter documentation")
    st.dataframe(results["parameter_documentation"], use_container_width=True)

with tab_5:
    st.subheader("Download results")

    excel_bytes = build_excel_bytes(results)

    file_name = (
        f"PaveLIFE_{input_dict_current['pavement_family']}_"
        f"{input_dict_current['pavement_type']}_{input_dict_current['shrp_id']}.xlsx"
    )

    st.download_button(
        label="Download Excel report",
        data=excel_bytes,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
