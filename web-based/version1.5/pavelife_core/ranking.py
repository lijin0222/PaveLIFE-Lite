"""Ranking utilities for PaveLIFE-Lite Web.

This module compares newly calculated PaveLIFE designs against an existing
case database of life-cycle GHG emissions and life-cycle costs.

Ranking convention:
    Lower GHG emissions or lower cost is better. The calculated design is
    inserted into the reference database, so a rank is reported as
    rank / (number of reference cases + 1). Ties share the same rank by using
    1 + count(reference values strictly lower than the calculated value).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


GHG_FILENAME = "GHG_emissions_all.xlsx"
COST_FILENAME = "Costs_all.xlsx"
PAVEMENT_TYPE_COL = "Pavement type"
GHG_TOTAL_COL = "Total Emission"
COST_TOTAL_COL = "Total Cost"


@dataclass(frozen=True)
class RankingDatabase:
    ghg: pd.DataFrame
    cost: pd.DataFrame
    ghg_path: str
    cost_path: str


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _standardize_pavement_type(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _read_reference_excel(path: str, total_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ranking reference file was not found: {path}")

    df = pd.read_excel(path)
    df = _clean_columns(df)

    required_cols = [PAVEMENT_TYPE_COL, total_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Ranking reference file {os.path.basename(path)} is missing required column(s): {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    df[PAVEMENT_TYPE_COL] = _standardize_pavement_type(df[PAVEMENT_TYPE_COL])
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce")
    df = df.dropna(subset=[PAVEMENT_TYPE_COL, total_col]).reset_index(drop=True)
    return df


def load_ranking_database(data_dir: str) -> RankingDatabase:
    """Load GHG and cost ranking databases from Excel files."""
    ghg_path = os.path.join(data_dir, GHG_FILENAME)
    cost_path = os.path.join(data_dir, COST_FILENAME)
    ghg_df = _read_reference_excel(ghg_path, GHG_TOTAL_COL)
    cost_df = _read_reference_excel(cost_path, COST_TOTAL_COL)
    return RankingDatabase(ghg=ghg_df, cost=cost_df, ghg_path=ghg_path, cost_path=cost_path)


def _rank_against_reference(value: float, reference_values: pd.Series) -> Dict[str, float]:
    """Return rank metadata against one reference population.

    Lower values rank better. The design itself is inserted into the reference
    population, so the denominator is N + 1.
    """
    value = pd.to_numeric(value, errors="coerce")
    reference_values = pd.to_numeric(reference_values, errors="coerce").dropna()
    n_reference = int(len(reference_values))

    if pd.isna(value) or n_reference == 0:
        return {
            "rank": np.nan,
            "denominator": np.nan,
            "reference_cases": n_reference,
            "percentile_from_best": np.nan,
            "better_than_percent": np.nan,
        }

    lower_count = int((reference_values < float(value)).sum())
    higher_count = int((reference_values > float(value)).sum())
    rank = lower_count + 1
    denominator = n_reference + 1
    percentile_from_best = rank / denominator * 100.0
    better_than_percent = higher_count / n_reference * 100.0 if n_reference else np.nan

    return {
        "rank": int(rank),
        "denominator": int(denominator),
        "reference_cases": int(n_reference),
        "percentile_from_best": float(percentile_from_best),
        "better_than_percent": float(better_than_percent),
    }


def _format_rank(rank, denominator) -> str:
    if pd.isna(rank) or pd.isna(denominator):
        return "N/A"
    return f"{int(rank)} / {int(denominator)}"


def _extract_reference_by_type(df: pd.DataFrame, pavement_type: str, total_col: str) -> pd.Series:
    pavement_type = str(pavement_type).strip().upper()
    typed_df = df[df[PAVEMENT_TYPE_COL] == pavement_type]
    return typed_df[total_col]


def build_ranking_summary(design_results: pd.DataFrame, ranking_db: RankingDatabase) -> pd.DataFrame:
    """Build ranking table for one or more calculated designs.

    Expected design_results columns:
        Design, Pavement type, Total GHG, kg CO2e, Total cost PV, USD
    """
    if design_results is None or len(design_results) == 0:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for _, row in design_results.iterrows():
        design = str(row.get("Design", "Design"))
        pavement_type = str(row.get("Pavement type", "")).strip().upper()
        ghg_value = pd.to_numeric(row.get("Total GHG, kg CO2e"), errors="coerce")
        cost_value = pd.to_numeric(row.get("Total cost PV, USD"), errors="coerce")

        ghg_all = _rank_against_reference(ghg_value, ranking_db.ghg[GHG_TOTAL_COL])
        ghg_type = _rank_against_reference(
            ghg_value,
            _extract_reference_by_type(ranking_db.ghg, pavement_type, GHG_TOTAL_COL),
        )
        cost_all = _rank_against_reference(cost_value, ranking_db.cost[COST_TOTAL_COL])
        cost_type = _rank_against_reference(
            cost_value,
            _extract_reference_by_type(ranking_db.cost, pavement_type, COST_TOTAL_COL),
        )

        rows.append(
            {
                "Design": design,
                "Pavement type": pavement_type,
                "Total GHG, kg CO2e": float(ghg_value) if pd.notna(ghg_value) else np.nan,
                "GHG all-database rank": _format_rank(ghg_all["rank"], ghg_all["denominator"]),
                "GHG all rank": ghg_all["rank"],
                "GHG all denominator": ghg_all["denominator"],
                "GHG all percentile from best, %": ghg_all["percentile_from_best"],
                "GHG better than all DB, %": ghg_all["better_than_percent"],
                "GHG same-type rank": _format_rank(ghg_type["rank"], ghg_type["denominator"]),
                "GHG type rank": ghg_type["rank"],
                "GHG type denominator": ghg_type["denominator"],
                "GHG type percentile from best, %": ghg_type["percentile_from_best"],
                "GHG better than same-type DB, %": ghg_type["better_than_percent"],
                "Total cost PV, USD": float(cost_value) if pd.notna(cost_value) else np.nan,
                "Cost all-database rank": _format_rank(cost_all["rank"], cost_all["denominator"]),
                "Cost all rank": cost_all["rank"],
                "Cost all denominator": cost_all["denominator"],
                "Cost all percentile from best, %": cost_all["percentile_from_best"],
                "Cost better than all DB, %": cost_all["better_than_percent"],
                "Cost same-type rank": _format_rank(cost_type["rank"], cost_type["denominator"]),
                "Cost type rank": cost_type["rank"],
                "Cost type denominator": cost_type["denominator"],
                "Cost type percentile from best, %": cost_type["percentile_from_best"],
                "Cost better than same-type DB, %": cost_type["better_than_percent"],
            }
        )

    return pd.DataFrame(rows)


def get_ranking_database_overview(ranking_db: RankingDatabase) -> pd.DataFrame:
    """Return compact database metadata for display."""
    ghg_types = ranking_db.ghg[PAVEMENT_TYPE_COL].value_counts().sort_index()
    cost_types = ranking_db.cost[PAVEMENT_TYPE_COL].value_counts().sort_index()
    all_types = sorted(set(ghg_types.index).union(set(cost_types.index)))
    rows = []
    for pavement_type in all_types:
        rows.append(
            {
                "Pavement type": pavement_type,
                "GHG reference cases": int(ghg_types.get(pavement_type, 0)),
                "Cost reference cases": int(cost_types.get(pavement_type, 0)),
            }
        )
    return pd.DataFrame(rows)
