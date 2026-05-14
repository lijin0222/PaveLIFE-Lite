# pavelife_core/schemas.py

from pavelife_core.config import valid_asphalt_types, valid_concrete_types


ui_to_notebook_key = {
    "shrp_id": "SHRP_ID",
    "pavement_family": "Pavement family",
    "pavement_type": "Pavement type",
    "iri0": "IRI0",
    "precipitation": "Precipitation",
    "temperature": "Temperature",
    "freeze_index": "Freeze index",
    "aadt": "AADT",
    "aadtt": "AADTT",
    "kesal": "KESAL",
    "surface_thickness": "Surface thickness",
    "base_thickness": "Base thickness",
    "subbase_thickness": "Subbase thickness",
    "func_class": "FUNC_CLASS",
}


def to_notebook_style_input(input_dict):
    """Convert Streamlit-style snake_case keys to original notebook input keys."""
    converted = {}

    for key, value in input_dict.items():
        target_key = ui_to_notebook_key.get(key, key)
        converted[target_key] = value

    if "Pavement type" in converted and converted["Pavement type"] is not None:
        converted["Pavement type"] = str(converted["Pavement type"]).upper().strip()

    if "Pavement family" in converted and converted["Pavement family"] is not None:
        converted["Pavement family"] = str(converted["Pavement family"]).lower().strip()

    return converted


def validate_section_input(input_dict):
    """Validate a single-section input dictionary and return notebook-style keys."""
    section_input = to_notebook_style_input(input_dict)

    required_keys = [
        "SHRP_ID",
        "Pavement family",
        "Pavement type",
        "IRI0",
        "Precipitation",
        "Temperature",
        "Freeze index",
        "AADT",
        "AADTT",
        "KESAL",
        "Surface thickness",
        "Base thickness",
        "Subbase thickness",
        "FUNC_CLASS",
    ]

    missing_keys = [key for key in required_keys if key not in section_input]
    if missing_keys:
        raise ValueError(f"Missing required input field(s): {missing_keys}")

    pavement_family = str(section_input["Pavement family"]).lower().strip()
    pavement_type = str(section_input["Pavement type"]).upper().strip()

    if pavement_family not in ["asphalt", "concrete"]:
        raise ValueError("Pavement family must be 'asphalt' or 'concrete'.")

    if pavement_family == "asphalt" and pavement_type not in valid_asphalt_types:
        raise ValueError(f"Asphalt pavement type must be one of {valid_asphalt_types}.")

    if pavement_family == "concrete" and pavement_type not in valid_concrete_types:
        raise ValueError(f"Concrete pavement type must be one of {valid_concrete_types}.")

    numeric_nonnegative_keys = [
        "Precipitation",
        "Freeze index",
        "AADT",
        "AADTT",
        "KESAL",
        "Surface thickness",
        "Base thickness",
        "Subbase thickness",
    ]

    for key in numeric_nonnegative_keys:
        value = float(section_input[key])
        if value < 0.0:
            raise ValueError(f"{key} must be non-negative.")
        section_input[key] = value

    section_input["Temperature"] = float(section_input["Temperature"])
    section_input["IRI0"] = float(section_input["IRI0"])

    if section_input["IRI0"] <= 0.0:
        raise ValueError("IRI0 must be greater than zero.")

    if float(section_input["AADTT"]) > float(section_input["AADT"]):
        raise ValueError("AADTT cannot be greater than AADT.")

    func_class = int(section_input["FUNC_CLASS"])
    if func_class < 1 or func_class > 8:
        raise ValueError("FUNC_CLASS must be an integer from 1 to 8.")
    section_input["FUNC_CLASS"] = func_class

    section_input["SHRP_ID"] = str(section_input["SHRP_ID"])
    section_input["Pavement family"] = pavement_family
    section_input["Pavement type"] = pavement_type

    return section_input
