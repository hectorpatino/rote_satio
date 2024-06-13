import pytest
import rioxarray
import xarray as xr
import numpy as np


@pytest.fixture(scope='module')
def test_array():

    band = 4
    y = 1000
    data = np.random.rand(band, y, 2)
    coords = {
        'band': np.arange(band),
        'y': np.arange(y),
        'x': np.arange(2)
    }
    dims = ('band', 'y', 'x')
    test_array = xr.DataArray(data, coords=coords, dims=dims)
    test_array.attrs['long_name'] = [np.arange(band)]
    test_array.rio.write_crs('EPSG:4326', inplace=True)
    return test_array

