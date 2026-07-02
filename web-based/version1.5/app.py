# app.py

import os
from io import BytesIO

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from pavelife_core.config import default_analysis_settings, func_class_lookup
from pavelife_core.model_loader import load_pretrained_rate_artifacts
from pavelife_core.outputs import build_excel_bytes
from pavelife_core.ranking import build_ranking_summary, get_ranking_database_overview, load_ranking_database
from pavelife_core.runner import run_pavelife_single_section


st.set_page_config(
    page_title="PaveLIFE-Lite",
    page_icon="🛣️",
    layout="wide",
)

if go is None:
    st.error("Plotly is required for this version of PaveLIFE-Lite Web. Please install it with: pip install plotly")
    st.stop()


plotly_config = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "pavelife_plot",
        "height": 900,
        "width": 1400,
        "scale": 3,
    },
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

pavelife_palette = [
    "#2A9D8F",
    "#E76F51",
    "#457B9D",
    "#F4A261",
    "#6D597A",
    "#8AB17D",
    "#B56576",
    "#577590",
    "#D4A373",
    "#264653",
]

lca_palette = ["#DDEDE9", "#B6DED5", "#87C9BB", "#55B3A4", "#2A9D8F", "#1E7F73"]
lcca_palette = ["#FCE4D6", "#F8C9A8", "#F3A76F", "#E88445", "#D96C2C", "#B9541F"]


def apply_custom_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1480px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F7FAFC 0%, #EEF4F8 100%);
            border-right: 1px solid #D9E2EC;
        }
        [data-testid="stSidebar"] h1 {
            color: #17324D;
            letter-spacing: -0.02em;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #23395D;
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.055);
        }
        div[data-testid="stMetric"] label {
            color: #5B677A;
            font-weight: 600;
        }
        .pavelife-hero {
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.0rem;
            border-radius: 24px;
            border: 1px solid #D7E4EE;
            background: linear-gradient(135deg, #F8FBFD 0%, #EAF4F4 48%, #F8F1E9 100%);
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        }
        .pavelife-hero h1 {
            margin-bottom: 0.2rem;
            color: #172A3A;
            font-size: 2.15rem;
            letter-spacing: -0.035em;
        }
        .pavelife-hero p {
            margin-top: 0.35rem;
            color: #425466;
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .pavelife-pill {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            margin-right: 0.35rem;
            margin-top: 0.45rem;
            border-radius: 999px;
            background: rgba(42, 157, 143, 0.13);
            color: #1F6F67;
            font-size: 0.82rem;
            font-weight: 650;
        }
        .pavelife-note {
            padding: 0.8rem 1rem;
            border-radius: 16px;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            color: #425466;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_css()


@st.cache_resource
def get_model_artifact(pavement_family, model_dir):
    """Load only the model artifact required by the selected pavement family."""
    return load_pretrained_rate_artifacts(
        pavement_family=pavement_family,
        model_dir=model_dir,
    )


def get_model_artifacts_for_families(pavement_families, model_dir):
    model_artifacts = {}
    for pavement_family in sorted(set(pavement_families)):
        model_artifacts[pavement_family] = get_model_artifact(pavement_family, model_dir)
    return model_artifacts


@st.cache_data(show_spinner=False)
def get_cached_ranking_database(data_dir):
    return load_ranking_database(data_dir)


def get_default_ranking_data_dir():
    return os.path.join(os.getcwd(), "data", "ranking_database")


def get_float_from_df(df, column_name, default_value=0.0):
    if isinstance(df, pd.DataFrame) and column_name in df.columns and len(df) > 0:
        value = pd.to_numeric(df[column_name].iloc[0], errors="coerce")
        if pd.notna(value):
            return float(value)
    return float(default_value)


def get_int_from_df(df, column_name, default_value=0):
    if isinstance(df, pd.DataFrame) and column_name in df.columns and len(df) > 0:
        value = pd.to_numeric(df[column_name].iloc[0], errors="coerce")
        if pd.notna(value):
            return int(value)
    return int(default_value)


func_class_label_to_code = {
    info["FUNC_CLASS_EXP"]: code for code, info in func_class_lookup.items()
}


def format_func_class_option(label):
    return label


def get_plot_layout(title=None, height=420):
    layout = dict(
        height=height,
        margin=dict(l=20, r=24, t=58 if title else 32, b=36),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=13, color="#334155"),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
    )
    if title:
        layout["title"] = dict(text=title, x=0.01, xanchor="left", font=dict(size=18, color="#172A3A"))
    return layout


def render_contribution_chart(contribution_df, label):
    if not isinstance(contribution_df, pd.DataFrame):
        return

    required_cols = ["Phase label", "Contribution rate (%)"]
    if not all(col in contribution_df.columns for col in required_cols):
        st.caption(f"{label} contribution chart skipped because expected columns were not found.")
        return

    chart_df = contribution_df[required_cols].copy()
    chart_df["Contribution rate (%)"] = pd.to_numeric(chart_df["Contribution rate (%)"], errors="coerce")
    chart_df = chart_df.dropna(subset=["Contribution rate (%)"])
    chart_df = chart_df.sort_values("Contribution rate (%)", ascending=True)

    palette = lca_palette if label.upper() == "LCA" else lcca_palette
    dominant_phase = ""
    if len(chart_df) > 0:
        dominant_phase = str(chart_df.loc[chart_df["Contribution rate (%)"].idxmax(), "Phase label"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["Contribution rate (%)"],
            y=chart_df["Phase label"],
            orientation="h",
            marker=dict(
                color=chart_df["Contribution rate (%)"],
                colorscale=palette,
                line=dict(color="rgba(255,255,255,0.9)", width=1.2),
            ),
            text=chart_df["Contribution rate (%)"].map(lambda x: f"{x:.1f}%"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2f}%<extra></extra>",
            name=label,
        )
    )
    fig.update_layout(
        **get_plot_layout(title=f"{label} contribution rate", height=390),
        showlegend=False,
        bargap=0.36,
        annotations=[
            dict(
                text=f"Dominant phase: <b>{dominant_phase}</b>" if dominant_phase else "",
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.08,
                showarrow=False,
                align="left",
                font=dict(size=12, color="#64748B"),
            )
        ],
    )
    fig.update_xaxes(
        title="Contribution rate (%)",
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        range=[0, max(100, float(chart_df["Contribution rate (%)"].max()) * 1.18 if len(chart_df) else 100)],
    )
    fig.update_yaxes(title="", showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


def get_iri_column(df_yearly):
    if "IRI_pred_before_maintenance" in df_yearly.columns:
        return "IRI_pred_before_maintenance", "IRI before maintenance used in LCA and LCCA"
    if "IRI_pred" in df_yearly.columns:
        return "IRI_pred", "IRI after maintenance used in LCA and LCCA"
    return None, None


def render_iri_chart(df_yearly):
    """Render only the IRI series actually used in LCA/LCCA calculations."""
    if not isinstance(df_yearly, pd.DataFrame):
        return

    iri_col, iri_label = get_iri_column(df_yearly)
    if iri_col is None:
        st.caption("IRI chart skipped because no IRI prediction column was found.")
        return

    if "Pavement_Age" not in df_yearly.columns:
        st.caption("IRI chart skipped because Pavement_Age was not found.")
        return

    chart_df = df_yearly[["Pavement_Age", iri_col]].copy()
    chart_df["Pavement_Age"] = pd.to_numeric(chart_df["Pavement_Age"], errors="coerce")
    chart_df[iri_col] = pd.to_numeric(chart_df[iri_col], errors="coerce")
    chart_df = chart_df.dropna(subset=["Pavement_Age", iri_col])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["Pavement_Age"],
            y=chart_df[iri_col],
            mode="lines",
            name=iri_label,
            line=dict(width=3.5, color="#2A9D8F", shape="spline", smoothing=0.55),
            fill="tozeroy",
            fillcolor="rgba(42,157,143,0.12)",
            hovertemplate="Age: %{x:.0f} years<br>IRI: %{y:.3f} m/km<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["Pavement_Age"],
            y=chart_df[iri_col],
            mode="markers",
            name="Annual prediction",
            marker=dict(size=5.5, color="#2A9D8F", line=dict(color="white", width=0.8)),
            hovertemplate="Age: %{x:.0f} years<br>IRI: %{y:.3f} m/km<extra></extra>",
        )
    )

    if "IRI_threshold" in df_yearly.columns:
        threshold_value = pd.to_numeric(df_yearly["IRI_threshold"].iloc[0], errors="coerce")
        if pd.notna(threshold_value):
            fig.add_hline(
                y=float(threshold_value),
                line=dict(color="#D94841", width=2.2, dash="dash"),
                annotation_text=f"IRI threshold = {float(threshold_value):.3f} m/km",
                annotation_position="top left",
                annotation_font=dict(color="#9B2C2C", size=12),
            )

    if "maintenance_flag" in df_yearly.columns:
        maintenance_df = df_yearly[df_yearly["maintenance_flag"] == 1].copy()
        if len(maintenance_df) > 0:
            maintenance_df["Pavement_Age"] = pd.to_numeric(maintenance_df["Pavement_Age"], errors="coerce")
            maintenance_df[iri_col] = pd.to_numeric(maintenance_df[iri_col], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=maintenance_df["Pavement_Age"],
                    y=maintenance_df[iri_col],
                    mode="markers",
                    name="M&R event",
                    marker=dict(symbol="diamond", size=12, color="#E76F51", line=dict(color="white", width=1.4)),
                    hovertemplate="M&R event<br>Age: %{x:.0f} years<br>IRI before treatment: %{y:.3f} m/km<extra></extra>",
                )
            )

    fig.update_layout(
        **get_plot_layout(title="IRI progression and threshold-triggered M&R", height=460),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )
    fig.update_xaxes(title="Pavement age (years)", showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(title="IRI (m/km)", showgrid=True, gridcolor="#E5E7EB", zeroline=False, rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    st.caption(f"Displayed IRI series: `{iri_col}`. This is the IRI series used by the LCA/LCCA usage-phase calculations.")


def render_comparison_iri_chart(comparison_items):
    fig = go.Figure()
    any_trace = False

    for idx, item in enumerate(comparison_items):
        df_yearly = item["results"].get("yearly_iri")
        if not isinstance(df_yearly, pd.DataFrame) or "Pavement_Age" not in df_yearly.columns:
            continue
        iri_col, _ = get_iri_column(df_yearly)
        if iri_col is None:
            continue
        color = pavelife_palette[idx % len(pavelife_palette)]
        chart_df = df_yearly[["Pavement_Age", iri_col]].copy()
        chart_df["Pavement_Age"] = pd.to_numeric(chart_df["Pavement_Age"], errors="coerce")
        chart_df[iri_col] = pd.to_numeric(chart_df[iri_col], errors="coerce")
        chart_df = chart_df.dropna(subset=["Pavement_Age", iri_col])
        fig.add_trace(
            go.Scatter(
                x=chart_df["Pavement_Age"],
                y=chart_df[iri_col],
                mode="lines",
                name=item["design_label"],
                line=dict(width=3, color=color, shape="spline", smoothing=0.45),
                hovertemplate=(
                    f"<b>{item['design_label']}</b><br>"
                    "Age: %{x:.0f} years<br>IRI: %{y:.3f} m/km<extra></extra>"
                ),
            )
        )
        any_trace = True

    if not any_trace:
        st.caption("Comparison IRI chart skipped because no valid IRI prediction column was found.")
        return

    first_df = comparison_items[0]["results"].get("yearly_iri")
    if isinstance(first_df, pd.DataFrame) and "IRI_threshold" in first_df.columns:
        threshold_value = pd.to_numeric(first_df["IRI_threshold"].iloc[0], errors="coerce")
        if pd.notna(threshold_value):
            fig.add_hline(
                y=float(threshold_value),
                line=dict(color="#D94841", width=2.0, dash="dash"),
                annotation_text=f"IRI threshold = {float(threshold_value):.3f} m/km",
                annotation_position="top left",
                annotation_font=dict(color="#9B2C2C", size=12),
            )

    fig.update_layout(
        **get_plot_layout(title="IRI progression comparison", height=500),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.76)",
        ),
    )
    fig.update_xaxes(title="Pavement age (years)", showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(title="IRI (m/km)", showgrid=True, gridcolor="#E5E7EB", zeroline=False, rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


def parse_maintenance_years(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value if pd.notna(x)]
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []
    years = []
    for part in text.replace(";", ",").split(","):
        part_clean = part.strip()
        if part_clean:
            try:
                years.append(int(float(part_clean)))
            except ValueError:
                pass
    return years


def get_maintenance_years_from_results(results):
    maintenance_summary = results.get("maintenance_summary")
    if isinstance(maintenance_summary, pd.DataFrame) and "maintenance_years" in maintenance_summary.columns and len(maintenance_summary) > 0:
        return parse_maintenance_years(maintenance_summary["maintenance_years"].iloc[0])
    if isinstance(maintenance_summary, pd.DataFrame) and "Maintenance Years" in maintenance_summary.columns and len(maintenance_summary) > 0:
        return parse_maintenance_years(maintenance_summary["Maintenance Years"].iloc[0])
    return []


def render_maintenance_timeline(comparison_items):
    rows = []
    for item in comparison_items:
        years = get_maintenance_years_from_results(item["results"])
        for year in years:
            rows.append({"Design": item["design_label"], "Maintenance year": year})

    if not rows:
        st.info("No maintenance events were triggered for the compared designs within the analysis period.")
        return

    timeline_df = pd.DataFrame(rows)
    fig = go.Figure()
    for idx, design in enumerate(timeline_df["Design"].unique()):
        sub_df = timeline_df[timeline_df["Design"] == design]
        color = pavelife_palette[idx % len(pavelife_palette)]
        fig.add_trace(
            go.Scatter(
                x=sub_df["Maintenance year"],
                y=sub_df["Design"],
                mode="markers",
                marker=dict(symbol="diamond", size=13, color=color, line=dict(color="white", width=1.2)),
                name=design,
                hovertemplate="<b>%{y}</b><br>M&R year: %{x}<extra></extra>",
            )
        )

    fig.update_layout(
        **get_plot_layout(title="Maintenance event timeline", height=max(300, 72 + 36 * len(timeline_df["Design"].unique()))),
        showlegend=False,
    )
    fig.update_xaxes(title="Pavement age (years)", showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(title="", showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


def render_metric_comparison_bar(summary_df, value_col, title, unit_label, color_scale):
    if not isinstance(summary_df, pd.DataFrame) or value_col not in summary_df.columns:
        return
    chart_df = summary_df[["Design", value_col]].copy()
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[value_col]).sort_values(value_col, ascending=True)
    if len(chart_df) == 0:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df[value_col],
            y=chart_df["Design"],
            orientation="h",
            marker=dict(
                color=chart_df[value_col],
                colorscale=color_scale,
                line=dict(color="rgba(255,255,255,0.9)", width=1.2),
            ),
            text=chart_df[value_col].map(lambda x: f"{x:,.0f}"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"<b>%{{y}}</b><br>{title}: %{{x:,.2f}} {unit_label}<extra></extra>",
        )
    )
    fig.update_layout(
        **get_plot_layout(title=title, height=max(360, 120 + 34 * len(chart_df))),
        showlegend=False,
        bargap=0.34,
    )
    fig.update_xaxes(title=unit_label, showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(title="", showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)




def build_single_ranking_source_df(results, input_dict_current):
    lca_summary = results.get("lca_summary")
    lcca_summary = results.get("lcca_summary")
    design_name = str(input_dict_current.get("shrp_id", "Current design"))
    pavement_type = str(input_dict_current.get("pavement_type", "")).upper()
    total_ghg = get_float_from_df(lca_summary, "Total Emission", 0.0)
    total_cost = get_float_from_df(lcca_summary, "Total Cost", 0.0)
    return pd.DataFrame(
        [
            {
                "Design": design_name,
                "Pavement type": pavement_type,
                "Total GHG, kg CO2e": total_ghg,
                "Total cost PV, USD": total_cost,
            }
        ]
    )


def get_ranking_summary_or_none(design_results_df):
    data_dir = get_default_ranking_data_dir()
    try:
        ranking_db = get_cached_ranking_database(data_dir)
        ranking_df = build_ranking_summary(design_results_df, ranking_db)
        overview_df = get_ranking_database_overview(ranking_db)
        return ranking_df, overview_df, None
    except Exception as error:
        return None, None, error


def _format_rank_value(rank_value):
    if rank_value is None or pd.isna(rank_value):
        return "N/A"
    return str(rank_value)


def _format_percent(value):
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}%"


def render_ranking_cards(ranking_df):
    if not isinstance(ranking_df, pd.DataFrame) or len(ranking_df) == 0:
        return

    if len(ranking_df) == 1:
        row = ranking_df.iloc[0]
        col_1, col_2, col_3, col_4 = st.columns(4)
        col_1.metric(
            "GHG rank in all DB",
            _format_rank_value(row.get("GHG all-database rank")),
            help="Lower life-cycle GHG emissions rank better. The calculated design is inserted into the reference database.",
        )
        col_2.metric(
            "GHG rank in same type",
            _format_rank_value(row.get("GHG same-type rank")),
            help="Ranking only among reference cases with the same pavement type.",
        )
        col_3.metric(
            "Cost rank in all DB",
            _format_rank_value(row.get("Cost all-database rank")),
            help="Lower life-cycle cost ranks better. The calculated design is inserted into the reference database.",
        )
        col_4.metric(
            "Cost rank in same type",
            _format_rank_value(row.get("Cost same-type rank")),
            help="Ranking only among reference cases with the same pavement type.",
        )
    else:
        col_1, col_2, col_3, col_4 = st.columns(4)
        best_ghg_all = ranking_df.loc[pd.to_numeric(ranking_df["GHG all rank"], errors="coerce").idxmin(), "Design"]
        best_ghg_type = ranking_df.loc[pd.to_numeric(ranking_df["GHG type rank"], errors="coerce").idxmin(), "Design"]
        best_cost_all = ranking_df.loc[pd.to_numeric(ranking_df["Cost all rank"], errors="coerce").idxmin(), "Design"]
        best_cost_type = ranking_df.loc[pd.to_numeric(ranking_df["Cost type rank"], errors="coerce").idxmin(), "Design"]
        col_1.metric("Best GHG rank, all DB", str(best_ghg_all))
        col_2.metric("Best GHG rank, same type", str(best_ghg_type))
        col_3.metric("Best cost rank, all DB", str(best_cost_all))
        col_4.metric("Best cost rank, same type", str(best_cost_type))


def render_ranking_percentile_chart(ranking_df, metric_label):
    if not isinstance(ranking_df, pd.DataFrame) or len(ranking_df) == 0:
        return

    if metric_label.upper() == "GHG":
        all_col = "GHG all percentile from best, %"
        type_col = "GHG type percentile from best, %"
        all_rank_col = "GHG all-database rank"
        type_rank_col = "GHG same-type rank"
        value_col = "Total GHG, kg CO2e"
        title = "GHG emissions ranking position"
        x_title = "Percentile from best rank (%)"
        colors = ["#2A9D8F", "#87C9BB"]
        value_suffix = "kg CO₂e"
    else:
        all_col = "Cost all percentile from best, %"
        type_col = "Cost type percentile from best, %"
        all_rank_col = "Cost all-database rank"
        type_rank_col = "Cost same-type rank"
        value_col = "Total cost PV, USD"
        title = "Life-cycle cost ranking position"
        x_title = "Percentile from best rank (%)"
        colors = ["#E76F51", "#F4A261"]
        value_suffix = "USD PV"

    plot_df = ranking_df[["Design", "Pavement type", value_col, all_col, type_col, all_rank_col, type_rank_col]].copy()
    for col in [value_col, all_col, type_col]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=[all_col, type_col])
    if len(plot_df) == 0:
        st.info(f"{metric_label} ranking chart is unavailable because no valid rank could be calculated.")
        return

    design_order = plot_df.sort_values(all_col, ascending=False)["Design"].tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df[all_col],
            y=plot_df["Design"],
            orientation="h",
            name="All database",
            marker=dict(color=colors[0], line=dict(color="white", width=1.0)),
            text=plot_df[all_rank_col],
            textposition="outside",
            cliponaxis=False,
            customdata=plot_df[["Pavement type", value_col, all_rank_col]],
            hovertemplate=(
                "<b>%{y}</b><br>Context: all database"
                "<br>Pavement type: %{customdata[0]}"
                f"<br>Value: %{{customdata[1]:,.2f}} {value_suffix}"
                "<br>Rank: %{customdata[2]}"
                "<br>Percentile from best: %{x:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=plot_df[type_col],
            y=plot_df["Design"],
            orientation="h",
            name="Same pavement type",
            marker=dict(color=colors[1], line=dict(color="white", width=1.0)),
            text=plot_df[type_rank_col],
            textposition="outside",
            cliponaxis=False,
            customdata=plot_df[["Pavement type", value_col, type_rank_col]],
            hovertemplate=(
                "<b>%{y}</b><br>Context: same pavement type"
                "<br>Pavement type: %{customdata[0]}"
                f"<br>Value: %{{customdata[1]:,.2f}} {value_suffix}"
                "<br>Rank: %{customdata[2]}"
                "<br>Percentile from best: %{x:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=50,
        line=dict(color="#94A3B8", width=1.2, dash="dot"),
        annotation_text="median position",
        annotation_position="top",
        annotation_font=dict(size=11, color="#64748B"),
    )
    fig.update_layout(
        **get_plot_layout(title=title, height=max(380, 160 + 42 * len(plot_df))),
        barmode="group",
        bargap=0.28,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        annotations=[
            dict(
                text="Lower percentile and smaller rank are better.",
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.08,
                showarrow=False,
                align="left",
                font=dict(size=12, color="#64748B"),
            )
        ],
    )
    fig.update_xaxes(title=x_title, range=[0, 105], showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(title="", categoryorder="array", categoryarray=design_order, showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


def render_ranking_tab(design_results_df):
    ranking_df, overview_df, error = get_ranking_summary_or_none(design_results_df)
    if error is not None:
        st.warning(
            "Ranking database could not be loaded. Make sure `Costs_all.xlsx` and `GHG_emissions_all.xlsx` "
            "are stored in `data/ranking_database/` under the project root."
        )
        st.exception(error)
        return None

    st.markdown(
        """
        <div class="pavelife-note">
        Ranking convention: lower GHG emissions or lower cost is better. The calculated design is inserted into the reference database, so ranks are shown as <b>rank / (reference cases + current design)</b>. Type-specific ranking uses only cases with the same pavement type.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_ranking_cards(ranking_df)

    st.subheader("GHG emissions ranking")
    render_ranking_percentile_chart(ranking_df, "GHG")

    st.subheader("Cost ranking")
    render_ranking_percentile_chart(ranking_df, "Cost")

    display_cols = [
        "Design",
        "Pavement type",
        "Total GHG, kg CO2e",
        "GHG all-database rank",
        "GHG same-type rank",
        "GHG better than all DB, %",
        "GHG better than same-type DB, %",
        "Total cost PV, USD",
        "Cost all-database rank",
        "Cost same-type rank",
        "Cost better than all DB, %",
        "Cost better than same-type DB, %",
    ]
    existing_cols = [col for col in display_cols if col in ranking_df.columns]
    st.subheader("Ranking summary table")
    st.dataframe(
        ranking_df[existing_cols].style.format(
            {
                "Total GHG, kg CO2e": "{:,.0f}",
                "Total cost PV, USD": "{:,.0f}",
                "GHG better than all DB, %": "{:.1f}%",
                "GHG better than same-type DB, %": "{:.1f}%",
                "Cost better than all DB, %": "{:.1f}%",
                "Cost better than same-type DB, %": "{:.1f}%",
            }
        ),
        use_container_width=True,
    )

    with st.expander("Reference database coverage by pavement type", expanded=False):
        st.dataframe(overview_df, use_container_width=True)
        st.caption("Reference files: `data/ranking_database/GHG_emissions_all.xlsx` and `data/ranking_database/Costs_all.xlsx`.")

    return ranking_df


def build_combined_contribution_df(comparison_items, contribution_key):
    rows = []
    for item in comparison_items:
        contribution_df = item["results"].get(contribution_key)
        if not isinstance(contribution_df, pd.DataFrame):
            continue
        if not all(col in contribution_df.columns for col in ["Phase label", "Contribution rate (%)"]):
            continue
        for _, row in contribution_df.iterrows():
            rows.append(
                {
                    "Design": item["design_label"],
                    "Phase": row["Phase label"],
                    "Contribution rate (%)": pd.to_numeric(row["Contribution rate (%)"], errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def render_comparison_contribution_stacked(comparison_items, contribution_key, title):
    contribution_df = build_combined_contribution_df(comparison_items, contribution_key)
    if len(contribution_df) == 0:
        st.caption(f"{title} skipped because no valid contribution table was found.")
        return
    phase_order = list(dict.fromkeys(contribution_df["Phase"].tolist()))
    design_order = list(dict.fromkeys([item["design_label"] for item in comparison_items]))

    fig = go.Figure()
    for idx, phase in enumerate(phase_order):
        sub_df = contribution_df[contribution_df["Phase"] == phase]
        value_map = dict(zip(sub_df["Design"], sub_df["Contribution rate (%)"]))
        x_values = [value_map.get(design, 0.0) for design in design_order]
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=design_order,
                orientation="h",
                name=phase,
                marker=dict(color=pavelife_palette[idx % len(pavelife_palette)], line=dict(color="white", width=0.8)),
                hovertemplate="<b>%{y}</b><br>Phase: " + phase + "<br>Contribution: %{x:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        **get_plot_layout(title=title, height=max(380, 140 + 35 * len(design_order))),
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.78)",
        ),
    )
    fig.update_xaxes(title="Contribution rate (%)", showgrid=True, gridcolor="#E5E7EB", zeroline=False, range=[0, 100])
    fig.update_yaxes(title="", showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


def build_common_inputs():
    shrp_id = st.text_input(
        "Section / scenario ID",
        value="demo_001",
        help="Shared section or scenario identifier.",
    )

    iri0 = st.number_input(
        "Initial IRI, m/km",
        min_value=0.01,
        value=0.95,
        step=0.05,
        help="Initial International Roughness Index shared by all compared designs.",
    )

    func_class_label = st.selectbox(
        "Functional class",
        options=list(func_class_label_to_code.keys()),
        index=0,
        format_func=format_func_class_option,
        help="The selected functional class determines the IRI trigger threshold and speed assumptions.",
    )
    func_class = func_class_label_to_code[func_class_label]

    func_info = func_class_lookup[int(func_class)]
    st.info(
        f"**{func_class_label}**\n\n"
        f"IRI threshold: {func_info['IRI_threshold']} m/km  \n"
        f"Car speed limit: {func_info['car_speed_limit_kph']} km/h  \n"
        f"Truck speed limit: {func_info['truck_speed_limit_kph']} km/h"
    )

    st.header("2. Shared climate and traffic")
    precipitation = st.number_input("Precipitation, mm", min_value=0.0, value=800.0, step=10.0)
    temperature = st.number_input("Temperature, °C", value=15.0, step=1.0)
    freeze_index = st.number_input("Freeze index, °C*days", min_value=0.0, value=100.0, step=10.0)
    aadt = st.number_input("AADT, vehicles/day", min_value=0.0, value=8000.0, step=100.0)
    aadtt = st.number_input("AADTT, trucks/day", min_value=0.0, value=800.0, step=50.0)
    kesal = st.number_input("KESAL", min_value=0.0, value=500.0, step=10.0)

    common_input = {
        "shrp_id": shrp_id,
        "iri0": iri0,
        "precipitation": precipitation,
        "temperature": temperature,
        "freeze_index": freeze_index,
        "aadt": aadt,
        "aadtt": aadtt,
        "kesal": kesal,
        "func_class": func_class,
    }
    return common_input


def build_design_input(prefix, default_label, default_family="asphalt", default_type=None, expanded=False):
    with st.expander(default_label, expanded=expanded):
        design_label = st.text_input("Design label", value=default_label, key=f"{prefix}_design_label")
        pavement_family = st.selectbox(
            "Pavement family",
            options=["asphalt", "concrete"],
            index=0 if default_family == "asphalt" else 1,
            key=f"{prefix}_pavement_family",
        )

        if pavement_family == "asphalt":
            pavement_type_options = ["ACUB", "ACATB", "ACCTB", "FDA", "IP"]
        else:
            pavement_type_options = ["JPCP", "JRCP", "CRCP"]

        if default_type in pavement_type_options:
            default_type_index = pavement_type_options.index(default_type)
        else:
            default_type_index = 0

        pavement_type = st.selectbox(
            "Pavement type",
            options=pavement_type_options,
            index=default_type_index,
            key=f"{prefix}_pavement_type",
        )

        st.caption("Input layer thicknesses directly in meters. PaveLIFE automatically converts them to inches only for the pretrained IRI deterioration model.")
        surface_thickness = st.number_input(
            "Surface thickness, m",
            min_value=0.0,
            value=0.15,
            step=0.01,
            format="%.3f",
            key=f"{prefix}_surface_thickness",
        )
        base_thickness = st.number_input(
            "Base thickness, m",
            min_value=0.0,
            value=0.25,
            step=0.01,
            format="%.3f",
            key=f"{prefix}_base_thickness",
        )
        subbase_thickness = st.number_input(
            "Subbase thickness, m",
            min_value=0.0,
            value=0.20,
            step=0.01,
            format="%.3f",
            key=f"{prefix}_subbase_thickness",
        )

    design_input = {
        "design_label": design_label,
        "pavement_family": pavement_family,
        "pavement_type": pavement_type,
        "surface_thickness": surface_thickness,
        "base_thickness": base_thickness,
        "subbase_thickness": subbase_thickness,
    }
    return design_input


def apply_expert_settings(analysis_settings, mode):
    if mode != "Expert mode":
        return analysis_settings

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
            st.number_input("Work-zone speed, km/h", min_value=1.0, value=float(default_analysis_settings["work_zone_speed_kph"]), step=5.0)
        )
        analysis_settings["work_hours_per_day"] = float(
            st.number_input("Work hours per day", min_value=1.0, value=float(default_analysis_settings["work_hours_per_day"]), step=1.0)
        )
        analysis_settings["overlay_only_duration_days"] = float(
            st.number_input("Overlay-only duration, days", min_value=0.0, value=float(default_analysis_settings["overlay_only_duration_days"]), step=1.0)
        )
        analysis_settings["mill_and_overlay_duration_days"] = float(
            st.number_input("Mill-and-overlay duration, days", min_value=0.0, value=float(default_analysis_settings["mill_and_overlay_duration_days"]), step=1.0)
        )
        analysis_settings["cpr_diamond_grinding_duration_days"] = float(
            st.number_input("CPR / diamond grinding duration, days", min_value=0.0, value=float(default_analysis_settings["cpr_diamond_grinding_duration_days"]), step=1.0)
        )
        analysis_settings["share_of_directional_traffic_during_work_hours"] = float(
            st.number_input(
                "Share of directional traffic during work hours",
                min_value=0.0,
                max_value=1.0,
                value=float(default_analysis_settings["share_of_directional_traffic_during_work_hours"]),
                step=0.01,
                format="%.3f",
            )
        )
        analysis_settings["peak_hour_multiplier"] = float(
            st.number_input("Peak-hour multiplier", min_value=0.0, value=float(default_analysis_settings["peak_hour_multiplier"]), step=0.05, format="%.2f")
        )
        analysis_settings["work_zone_capacity_per_open_lane_vehph"] = float(
            st.number_input(
                "Work-zone capacity per open lane, veh/h/lane",
                min_value=1.0,
                value=float(default_analysis_settings["work_zone_capacity_per_open_lane_vehph"]),
                step=50.0,
            )
        )
        analysis_settings["mr_open_lanes"] = int(
            st.number_input("Open lanes during M&R", min_value=1, max_value=20, value=int(default_analysis_settings["mr_open_lanes"]), step=1)
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
            st.number_input(
                "Reinforcement ratio for JRCP/CRCP",
                min_value=0.0,
                max_value=0.20,
                value=float(default_analysis_settings["reinforcement_ratio"]),
                step=0.001,
                format="%.4f",
            )
        )

    return analysis_settings


def build_sidebar_inputs():
    with st.sidebar:
        st.title("PaveLIFE-Lite")
        st.caption("Single-section and design-comparison pavement LCA-LCCA decision-support tool")

        analysis_mode = st.radio(
            "Analysis mode",
            options=["Single pavement design", "Design comparison"],
            index=0,
            help="Single design evaluates one pavement structure. Design comparison evaluates up to 10 alternatives under the same climate, traffic, and analysis assumptions.",
        )

        input_mode = st.radio(
            "Input mode",
            options=["Basic mode", "Expert mode"],
            index=0,
            help="Basic mode exposes core inputs. Expert mode exposes selected LCA/LCCA assumptions.",
        )

        st.divider()
        st.header("1. Shared section context")
        common_input = build_common_inputs()

        if analysis_mode == "Single pavement design":
            st.header("3. Pavement design")
            design_input = build_design_input(
                prefix="single",
                default_label="Single design",
                default_family="asphalt",
                default_type="ACUB",
                expanded=True,
            )
            design_inputs = [design_input]
        else:
            st.header("3. Design alternatives")
            st.caption("All designs share the same IRI0, functional class, climate, traffic, and analysis assumptions. Only pavement family, pavement type, and layer thicknesses vary by design.")
            number_of_designs = st.number_input(
                "Number of designs to compare",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
            )
            design_inputs = []
            default_types = ["ACUB", "ACATB", "JPCP", "JRCP", "CRCP", "FDA", "IP", "ACCTB", "ACUB", "JPCP"]
            for idx in range(int(number_of_designs)):
                default_type = default_types[idx % len(default_types)]
                default_family = "concrete" if default_type in ["JPCP", "JRCP", "CRCP"] else "asphalt"
                design_inputs.append(
                    build_design_input(
                        prefix=f"comparison_{idx + 1}",
                        default_label=f"Design {idx + 1}",
                        default_family=default_family,
                        default_type=default_type,
                        expanded=(idx < 2),
                    )
                )

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

        analysis_settings = default_analysis_settings.copy()
        analysis_settings["analysis_years"] = int(analysis_years)
        analysis_settings["analysis_period_years"] = int(analysis_years)
        analysis_settings["real_discount_rate"] = float(real_discount_rate)
        analysis_settings["thickness_unit_to_m"] = 1.0
        analysis_settings = apply_expert_settings(analysis_settings, input_mode)

        run_button = st.button("Run PaveLIFE", type="primary", use_container_width=True)

    return analysis_mode, input_mode, common_input, design_inputs, analysis_settings, model_dir, run_button


def build_input_dict(common_input, design_input, design_index=1):
    shrp_id = common_input["shrp_id"]
    design_label = design_input.get("design_label", f"Design {design_index}")
    input_dict = {
        "shrp_id": f"{shrp_id}_{design_label}" if design_label else shrp_id,
        "pavement_family": design_input["pavement_family"],
        "pavement_type": design_input["pavement_type"],
        "iri0": common_input["iri0"],
        "precipitation": common_input["precipitation"],
        "temperature": common_input["temperature"],
        "freeze_index": common_input["freeze_index"],
        "aadt": common_input["aadt"],
        "aadtt": common_input["aadtt"],
        "kesal": common_input["kesal"],
        "surface_thickness": design_input["surface_thickness"],
        "base_thickness": design_input["base_thickness"],
        "subbase_thickness": design_input["subbase_thickness"],
        "func_class": common_input["func_class"],
    }
    return input_dict


def run_single_design(input_dict, analysis_settings, model_dir):
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
    return results


def run_design_comparison(common_input, design_inputs, analysis_settings, model_dir):
    pavement_families = [design_input["pavement_family"] for design_input in design_inputs]
    model_artifacts = get_model_artifacts_for_families(pavement_families, model_dir)
    comparison_items = []
    progress_bar = st.progress(0.0)

    for idx, design_input in enumerate(design_inputs):
        input_dict = build_input_dict(common_input, design_input, design_index=idx + 1)
        results = run_pavelife_single_section(
            input_dict=input_dict,
            model_artifacts=model_artifacts,
            analysis_settings=analysis_settings,
            model_dir=model_dir,
            output_dir=os.path.join(os.getcwd(), "outputs"),
            make_figures=False,
            save_excel_outputs=False,
        )
        design_label = design_input.get("design_label", f"Design {idx + 1}")
        if not design_label:
            design_label = f"Design {idx + 1}"
        comparison_items.append(
            {
                "design_label": design_label,
                "input_dict": input_dict,
                "design_input": design_input,
                "results": results,
            }
        )
        progress_bar.progress((idx + 1) / len(design_inputs))

    progress_bar.empty()
    return comparison_items


def build_comparison_summary(comparison_items):
    rows = []
    for item in comparison_items:
        results = item["results"]
        input_dict = item["input_dict"]
        rate_prediction = results.get("rate_prediction")
        maintenance_summary = results.get("maintenance_summary")
        lca_summary = results.get("lca_summary")
        lcca_summary = results.get("lcca_summary")
        maintenance_years = get_maintenance_years_from_results(results)

        rows.append(
            {
                "Design": item["design_label"],
                "Pavement family": input_dict["pavement_family"],
                "Pavement type": input_dict["pavement_type"],
                "Surface thickness, m": input_dict["surface_thickness"],
                "Base thickness, m": input_dict["base_thickness"],
                "Subbase thickness, m": input_dict["subbase_thickness"],
                "rate_log_pred": get_float_from_df(rate_prediction, "rate_log_pred", 0.0),
                "rate_pred": get_float_from_df(rate_prediction, "rate_pred", 0.0),
                "Maintenance count": get_int_from_df(maintenance_summary, "maintenance_count", len(maintenance_years)),
                "First maintenance year": maintenance_years[0] if maintenance_years else "",
                "Maintenance years": ", ".join(str(year) for year in maintenance_years),
                "Total GHG, kg CO2e": get_float_from_df(lca_summary, "Total Emission", 0.0),
                "Total cost PV, USD": get_float_from_df(lcca_summary, "Total Cost", 0.0),
            }
        )
    return pd.DataFrame(rows)


def build_comparison_excel_bytes(comparison_items, comparison_summary_df, ranking_df=None):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        comparison_summary_df.to_excel(writer, sheet_name="comparison_summary", index=False)
        if isinstance(ranking_df, pd.DataFrame) and len(ranking_df) > 0:
            ranking_df.to_excel(writer, sheet_name="ranking_summary", index=False)
        for idx, item in enumerate(comparison_items, start=1):
            label = str(item["design_label"])
            safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)[:12]
            results = item["results"]
            sheet_prefix = f"D{idx}_{safe_label}"
            for key in ["input_parameters", "rate_prediction", "yearly_iri", "maintenance_summary", "lca_summary", "lca_phase_contribution", "lcca_summary", "lcca_phase_contribution"]:
                value = results.get(key)
                if isinstance(value, pd.DataFrame):
                    sheet_name = f"{sheet_prefix}_{key}"[:31]
                    value.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


def render_single_results(results, input_dict_current, input_mode):
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

    st.caption(f"Analysis mode: Single pavement design. Input mode: {input_mode}. Predicted rate_log: {rate_log_pred:.6f}.")

    ranking_source_df = build_single_ranking_source_df(results, input_dict_current)

    tab_1, tab_2, tab_3, tab_4, tab_5, tab_6 = st.tabs(["IRI & M&R", "LCA", "LCCA", "Ranking", "Model and assumptions", "Download"])

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
        st.subheader("Ranking against existing case database")
        ranking_df = render_ranking_tab(ranking_source_df)

    with tab_5:
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

    with tab_6:
        st.subheader("Download results")
        results_for_export = results.copy()
        if 'ranking_df' in locals() and isinstance(ranking_df, pd.DataFrame):
            results_for_export["ranking_summary"] = ranking_df
        excel_bytes = build_excel_bytes(results_for_export)
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


def render_comparison_results(comparison_items, input_mode):
    comparison_summary_df = build_comparison_summary(comparison_items)

    best_ghg_idx = comparison_summary_df["Total GHG, kg CO2e"].astype(float).idxmin()
    best_cost_idx = comparison_summary_df["Total cost PV, USD"].astype(float).idxmin()
    min_ghg_design = str(comparison_summary_df.loc[best_ghg_idx, "Design"])
    min_cost_design = str(comparison_summary_df.loc[best_cost_idx, "Design"])

    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Compared designs", f"{len(comparison_items)}")
    col_2.metric("Lowest GHG design", min_ghg_design)
    col_3.metric("Lowest cost design", min_cost_design)
    col_4.metric("Input mode", input_mode)

    st.caption("Analysis mode: Design comparison. Climate, traffic, functional class, IRI0, and analysis settings are shared across all compared designs.")

    tab_1, tab_2, tab_3, tab_4, tab_5, tab_6 = st.tabs(["Comparison summary", "IRI & M&R", "LCA", "LCCA", "Ranking", "Download"])

    with tab_1:
        st.subheader("Design comparison summary")
        st.dataframe(comparison_summary_df, use_container_width=True)

    with tab_2:
        st.subheader("IRI progression comparison")
        render_comparison_iri_chart(comparison_items)
        st.subheader("Maintenance timeline")
        render_maintenance_timeline(comparison_items)
        with st.expander("Yearly IRI tables by design", expanded=False):
            for item in comparison_items:
                st.markdown(f"**{item['design_label']}**")
                st.dataframe(item["results"]["yearly_iri"], use_container_width=True)

    with tab_3:
        st.subheader("Total life-cycle GHG emissions")
        render_metric_comparison_bar(
            comparison_summary_df,
            "Total GHG, kg CO2e",
            "Total life-cycle GHG emissions",
            "kg CO₂e",
            lca_palette,
        )
        st.subheader("LCA contribution structure")
        render_comparison_contribution_stacked(comparison_items, "lca_phase_contribution", "LCA contribution rate by design")
        with st.expander("LCA tables by design", expanded=False):
            for item in comparison_items:
                st.markdown(f"**{item['design_label']}**")
                st.dataframe(item["results"]["lca_summary"], use_container_width=True)
                st.dataframe(item["results"]["lca_phase_contribution"], use_container_width=True)

    with tab_4:
        st.subheader("Total life-cycle cost")
        render_metric_comparison_bar(
            comparison_summary_df,
            "Total cost PV, USD",
            "Total life-cycle cost PV",
            "USD PV",
            lcca_palette,
        )
        st.subheader("LCCA contribution structure")
        render_comparison_contribution_stacked(comparison_items, "lcca_phase_contribution", "LCCA contribution rate by design")
        with st.expander("LCCA tables by design", expanded=False):
            for item in comparison_items:
                st.markdown(f"**{item['design_label']}**")
                st.dataframe(item["results"]["lcca_summary"], use_container_width=True)
                st.dataframe(item["results"]["lcca_phase_contribution"], use_container_width=True)

    with tab_5:
        st.subheader("Ranking against existing case database")
        ranking_df = render_ranking_tab(comparison_summary_df)

    with tab_6:
        st.subheader("Download comparison results")
        if 'ranking_df' in locals() and isinstance(ranking_df, pd.DataFrame):
            comparison_summary_for_export = comparison_summary_df.copy()
            excel_bytes = build_comparison_excel_bytes(comparison_items, comparison_summary_for_export, ranking_df=ranking_df)
        else:
            excel_bytes = build_comparison_excel_bytes(comparison_items, comparison_summary_df)
        st.download_button(
            label="Download comparison Excel report",
            data=excel_bytes,
            file_name="PaveLIFE_design_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


st.markdown(
    """
    <div class="pavelife-hero">
        <h1>PaveLIFE-Lite Web</h1>
        <p>
        An interactive pavement LCA-LCCA tool integrating pretrained IRI deterioration-rate prediction,
        threshold-triggered maintenance simulation, life-cycle GHG accounting, life-cycle cost analysis,
        and multi-design comparison under shared climate and traffic scenarios.
        </p>
        <span class="pavelife-pill">Single design</span>
        <span class="pavelife-pill">Design comparison</span>
        <span class="pavelife-pill">IRI prediction</span>
        <span class="pavelife-pill">LCA</span>
        <span class="pavelife-pill">LCCA</span>
        <span class="pavelife-pill">M&R simulation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

analysis_mode, input_mode, common_input, design_inputs, analysis_settings, model_dir, run_button = build_sidebar_inputs()

if "pavelife_results_payload" not in st.session_state:
    st.session_state["pavelife_results_payload"] = None

if not run_button and st.session_state["pavelife_results_payload"] is None:
    st.info("Select an analysis mode, enter parameters in the sidebar, and click **Run PaveLIFE**.")
    st.stop()

if run_button:
    try:
        with st.spinner("Running PaveLIFE calculation..."):
            if analysis_mode == "Single pavement design":
                input_dict = build_input_dict(common_input, design_inputs[0], design_index=1)
                results = run_single_design(input_dict, analysis_settings, model_dir)
                st.session_state["pavelife_results_payload"] = {
                    "analysis_mode": analysis_mode,
                    "input_mode": input_mode,
                    "input_dict": input_dict,
                    "results": results,
                }
            else:
                comparison_items = run_design_comparison(common_input, design_inputs, analysis_settings, model_dir)
                st.session_state["pavelife_results_payload"] = {
                    "analysis_mode": analysis_mode,
                    "input_mode": input_mode,
                    "comparison_items": comparison_items,
                }
        st.success("PaveLIFE calculation completed.")

    except Exception as error:
        st.error("PaveLIFE calculation failed.")
        st.exception(error)
        st.stop()

payload = st.session_state["pavelife_results_payload"]

if payload["analysis_mode"] == "Single pavement design":
    render_single_results(
        results=payload["results"],
        input_dict_current=payload["input_dict"],
        input_mode=payload["input_mode"],
    )
else:
    render_comparison_results(
        comparison_items=payload["comparison_items"],
        input_mode=payload["input_mode"],
    )
