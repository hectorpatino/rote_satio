import pytest
import xarray as xr

from rote_satio.models.segmentatition import AutoSegmentation
from rote_satio.utils.geobia_transformer import SegmentsTransformer
from rote_satio.utils.index_transformer import IndexTransformer

transformers = [SegmentsTransformer(), IndexTransformer()]


@pytest.mark.parametrize("transformer", transformers)
def test_index_transformer_inputs(transformer):
    it = IndexTransformer()
    with pytest.raises(TypeError):
        it.transform([1, 2, 3])


@pytest.mark.parametrize("transformer", transformers)
def test_index_transformer_output(transformer, test_array):
    foo = transformer.transform(test_array)
    assert type(foo) is xr.DataArray


def test_segmentation_transformer(test_array):
    transformer = AutoSegmentation(image_program='Planet')
    foo = transformer.predict(test_array)
    assert type(foo) is xr.DataArray


def test_segmentation_transformer_train(test_array):
    transformer = AutoSegmentation(image_program='Planet', model='TDD')
    foo = transformer.predict(test_array)
    assert type(foo) is xr.DataArray





