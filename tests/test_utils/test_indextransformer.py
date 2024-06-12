import pytest
import xarray as xr

from rote_satio.utils.index_transformer import IndexTransformer


def test_index_transformer_inputs():
    it = IndexTransformer()
    with pytest.raises(TypeError):
        it.transform([1, 2, 3])


def test_index_transformer_output(df_xarray):
    it = IndexTransformer()
    foo = it.transform(df_xarray)
    assert type(foo) is xr.DataArray
