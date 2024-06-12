import pytest
import rioxarray
import xarray as xr
import numpy as np

@pytest.fixture(scope='module')
def df_xarray():


    data = np.random.rand(4, 3, 2)
    coords = {
        'band': np.arange(4),
        'lat': np.arange(3),
        'lon': np.arange(2)
    }
    dims = ('band', 'lat', 'lon')
    test_array = xr.DataArray(data, coords=coords, dims=dims)
    test_array.attrs['long_name'] = [np.arange(4)]
    test_array.rio.write_crs('EPSG:4326', inplace=True)
    return test_array

