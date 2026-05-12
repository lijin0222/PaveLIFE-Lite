# pavelife_core/config.py

from pavelife_core import legacy

valid_asphalt_types = legacy.valid_asphalt_types
valid_concrete_types = legacy.valid_concrete_types
valid_pavement_types = legacy.valid_pavement_types

asphalt_type_config = legacy.asphalt_type_config
concrete_type_config = legacy.concrete_type_config
pavement_type_to_family = legacy.pavement_type_to_family
func_class_lookup = legacy.func_class_lookup

default_analysis_settings = {
    "analysis_years": legacy.analysis_years,
    "analysis_base_year": legacy.analysis_base_year,
    "analysis_period_years": legacy.analysis_period_years,
    "real_discount_rate": legacy.real_discount_rate,
    "thickness_unit_to_m": legacy.thickness_unit_to_m,
    "growth_rate": legacy.growth_rate,
    "lane_length": legacy.lane_length,
    "lane_width": legacy.lane_width,
    "work_zone_speed_kph": legacy.work_zone_speed_kph,
    "work_hours_per_day": legacy.work_hours_per_day,
    "overlay_only_duration_days": legacy.overlay_only_duration_days,
    "mill_and_overlay_duration_days": legacy.mill_and_overlay_duration_days,
    "cpr_diamond_grinding_duration_days": legacy.cpr_diamond_grinding_duration_days,
    "gasoline_price_per_gallon": legacy.gasoline_price_per_gallon,
    "diesel_price_per_gallon": legacy.diesel_price_per_gallon,
    "asphalt_prod_factor": legacy.asphalt_prod_factor,
    "concrete_prod_factor": legacy.concrete_prod_factor,
    "aggregate_prod_factor": legacy.aggregate_prod_factor,
    "ctb_prod_factor": legacy.ctb_prod_factor,
    "atb_prod_factor": legacy.atb_prod_factor,
    "rebar_prod_factor": legacy.rebar_prod_factor,
    "asphalt_material_cost_per_ton": legacy.asphalt_material_cost_per_ton,
    "concrete_material_cost_per_ton": legacy.concrete_material_cost_per_ton,
    "aggregate_material_cost_per_ton": legacy.aggregate_material_cost_per_ton,
    "reinforcement_ratio": legacy.reinforcement_ratio,
}
