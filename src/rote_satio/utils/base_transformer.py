import xarray as xr
from sklearn.base import TransformerMixin, BaseEstimator


class BaseIOTransformer(BaseEstimator, TransformerMixin):
    def _check_input(self, X):
        if not isinstance(X, xr.DataArray):
            raise TypeError(f"X should be of type xarray.DataArray. Got {type(X)}")

    def _check_program(self, program):
        programs = ["Planet", "Landsat-TM", "MODIS", "Sentinel-2", "Landsat-ETM+", "Landsat-OLI"]
        if program not in programs:
            raise ValueError(f"Invalid image program {program}. Supported programs are {programs}.")
