import xarray as xr
import numpy as np
import rioxarray
from skimage.segmentation import quickshift
import spyndex
import scipy

xds = rioxarray.open_rasterio('../data/2018-12_11001_01.tif')
NDVI = spyndex.computeIndex("NDVI",{"N": xds.sel(band = 4),"R": xds.sel(band = 1)})

#calcuar segmentos
segments = quickshift(NDVI.data,
                      kernel_size=1,
                      convert2lab=False,
                      max_dist=2,
                      ratio=1.0)
segments_zonal_mean_qs = scipy.ndimage.mean(input=NDVI.data,
                                            labels=segments,
                                            index=segments)
# añadir una dimensión. (1, 524, 524)
segments_zonal_mean_qs = np.expand_dims(segments_zonal_mean_qs, axis=0)


zeros = np.zeros((1, 524, 524))
work_data = xds.data

work_data = np.concatenate([work_data, zeros], axis=0)
work_data[work_data.shape[0]-1] = NDVI.data


work_data = np.concatenate([work_data, zeros], axis=0)
work_data[work_data.shape[0]-1] = segments_zonal_mean_qs

work_data_dataset = xr.DataArray(
    work_data,
    dims = ['band', 'y', 'x'],
    coords = {
        'band': ['B1', 'B2', 'B3', 'B4', 'NDVI', 'ZonalMeanQS'],
        'y': xds.coords['y'],
        'x': xds.coords['x']
    }

)
work_data_dataset = work_data_dataset.rio.write_crs(xds.rio.crs)
work_data_dataset.rio.to_raster("../data/example.tif")
