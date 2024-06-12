import pytest
import xarray as xr

from rote_satio.utils.geobia_transformer import SegmentsTransformer
from rote_satio.utils.index_transformer import IndexTransformer

transformers = [SegmentsTransformer(), IndexTransformer()]


@pytest.mark.parametrize("transformer", transformers)
def test_index_transformer_inputs(transformer, df_xarray):
    it = IndexTransformer()
    with pytest.raises(TypeError):
        it.transform([1, 2, 3])


@pytest.mark.parametrize("transformer", transformers)
def test_index_transformer_output(transformer, df_xarray):
    foo = transformer.transform(df_xarray)
    assert type(foo) is xr.DataArray
