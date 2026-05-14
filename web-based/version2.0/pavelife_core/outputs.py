# pavelife_core/outputs.py

from io import BytesIO

import pandas as pd

from pavelife_core.legacy import (
    build_phase_contribution_df,
    build_layer_stage_table,
    build_parameter_documentation_df,
    build_rate_metadata_df,
    build_pavement_type_config_df,
    save_combined_outputs,
)


def build_excel_bytes(results):
    """Build an in-memory Excel workbook for Streamlit download."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, value in results.items():
            if isinstance(value, pd.DataFrame):
                safe_sheet_name = str(sheet_name)[:31]
                value.to_excel(writer, sheet_name=safe_sheet_name, index=False)

    output.seek(0)
    return output.getvalue()
