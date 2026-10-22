import pytest

from pseudopeople.schema_entities import COLUMNS, DATASETS
from tests.integration.conftest import _get_common_datasets, _load_sample_data


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.ssa.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        DATASETS.tax_1040.name,
    ],
)
def test_unnoised_id_cols(dataset_name: str, request):
    """Tests that all datasets retain unnoised simulant_id and household_id
    (except for SSA which does not include household_id)
    """
    # simulant_id is part of the key the noised and unnoised data are matched on
    if dataset_name == DATASETS.ssa.name:
        pytest.skip(reason="SSA only has simulant_id, which is the matching key")
    data = _load_sample_data(dataset_name, request)
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    check_noised, check_original = _get_common_datasets(dataset_name, data, noised_data)
    assert len(check_noised) > 0
    household_id = COLUMNS.household_id.name
    assert check_original[household_id].equals(check_noised[household_id])
