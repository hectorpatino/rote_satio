import pytest
import xarray as xr
from rote_satio.utils.geobia_transformer import SegmentsTransformer


def test_segments_transformer_inputs():
    st = SegmentsTransformer()
    with pytest.raises(TypeError):
        st.transform([1, 2, 3])

def test_segments_transformer_output(df_xarray):
    transformer = SegmentsTransformer()
    foo = transformer.transform(df_xarray)
    assert type(foo) is xr.DataArray
