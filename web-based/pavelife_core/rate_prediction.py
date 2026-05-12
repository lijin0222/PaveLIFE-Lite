# pavelife_core/rate_prediction.py

from pavelife_core.legacy import (
    safe_file_name,
    normalize_column_name,
    infer_pavement_family,
    validate_section_input as legacy_validate_section_input,
    build_single_section_dataframe,
    get_encoder_input_columns,
    get_encoder_feature_names,
    get_value_for_model_column,
    build_numeric_prediction_features,
    build_categorical_prediction_features,
    transform_rate_features,
    predict_rate_for_single_section,
)
