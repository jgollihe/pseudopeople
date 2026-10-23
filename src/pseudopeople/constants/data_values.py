from pseudopeople.constants.metadata import DatasetNames

# Targeted omission constants for do_not_respond
DO_NOT_RESPOND_BASE_PROBABILITY = 0.0024

DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_RACE: dict[str, float] = {
    "AIAN": 0.0067,
    "Asian": -0.0286,
    "Black": 0.0306,
    "Latino": 0.0475,
    "Multiracial or Other": 0.041,
    "NHOPI": -0.0152,
    "White": -0.0188,
}

# Age bins include their lower bound and exclude their upper bound
DO_NOT_RESPOND_AGE_BIN_EDGES: list[int] = [0, 5, 10, 18, 30, 50, 125]

DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_SEX_AGE: dict[str, list[float]] = {
    "Female": [0.0255, -0.0014, -0.0003, 0.0074, -0.0034, -0.0287],
    "Male": [0.0255, -0.0014, -0.0003, 0.0201, 0.0281, -0.0079],
}

DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY: dict[str, float] = {
    DatasetNames.ACS: 0.0145,  # 1.45%
    DatasetNames.CPS: 0.2905,  # 29.05%
    DatasetNames.CENSUS: 0.0145,  # 1.45%
}
