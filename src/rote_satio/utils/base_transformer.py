import xarray as xr
from sklearn.base import TransformerMixin, BaseEstimator


class BaseIOTransformer(BaseEstimator, TransformerMixin):
    def _check_input(self, X):
        if not isinstance(X, xr.DataArray):
            raise TypeError(f"X should be of type xarray.DataArray. Got {type(X)}")
