import os
import scipy
import rioxarray

import xarray as xr
import numpy as np

from kneed import KneeLocator
from typing import Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.wrappers import SklearnTransformerWrapper

from skimage.segmentation import quickshift
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from rote_satio.models.segmentatition.planet.dispatcher import PlanetPipeline
from rote_satio.utils.base_transformer import BaseIOTransformer
from rote_satio.utils.geobia_transformer import SegmentsTransformer
from rote_satio.utils.index_transformer import IndexTransformer
from rote_satio.utils.utils import parse_to_pandas


class AutoSegmentation(BaseIOTransformer):
    def __init__(
            self,
            image_program: str,
            model: str = 'basic',
            image_path: Optional[str] = None,
            generate_objects: bool = True,
    ):
        """
        Args:
            image_path: Path to the image that needs to be segmented.
            image_program: Program of the sensor of the image.
            generate_objects: If True, there will be generation of geobias objects.

        Returns:
            self (AutoSegmentation): And AutoSegmentation object with the given image_path and image_program
                that can be used to perform segmentation on the image.
        """
        self.image_path = image_path
        self.image_program = image_program
        self.generate_objects = generate_objects
        self.umap_pipeline = None
        self.hdbscan_pipeline = None
        # TODO qas
        self.model = model
        self._check_transformer_input()

    def predict(
            self,
            X: xr.DataArray = None,
            range_values: Tuple[int, int] = (5, 50),
    ) -> xr.DataArray:
        """
        Predicts the segmentation of the image. Using a pre-trained model.
        Returns:
            xr.DataArray: DataArray of the predicted labels.

        """
        if self.image_path is not None and X is not None:
            raise ValueError("Image path and data cannot be None at the same time.")
        elif self.image_path is None and X is not None:
            self.data = X
        elif self.image_path is not None:
            self.data = self._read_image()
        else:
            raise ValueError("Image path or data must be provided.")
        self.data = self._generate_indexes()
        if self.generate_objects:
            self.data = self._generete_segments()
        df = parse_to_pandas(self.data)
        if self.model == 'basic':
            return self._train(values=range_values)
        else:
            pipeline = PlanetPipeline()
            labels = pipeline.fit_predict(df)
        return self._convert_to_xarray(labels)


    def _train(self, values):
        """
        Trains a models to segment the image. It uses kmeans to segment the image and uses
        the elbow method to find the best number of clusters.
        Args:
            values: A tuple with the range of number of clusters to test.

        Returns:
            self (AutoSegmentation): DataArray of the predicted labels.

        """
        df = parse_to_pandas(self.data)
        number_clusters = np.arange(values[0], values[1])
        pipeline = Pipeline([
            ('scaler', SklearnTransformerWrapper(transformer=MinMaxScaler())),
            ('robust_scaler', SklearnTransformerWrapper(transformer=RobustScaler())),
            ('scaler_2', SklearnTransformerWrapper(transformer=MinMaxScaler())),
            ('discretizer', EqualWidthDiscretiser(bins=10)),
            ('scaler_3', SklearnTransformerWrapper(transformer=MinMaxScaler())),
        ])
        df = pipeline.fit_transform(df)

        inertia_list = []

        for num_clust in number_clusters:
            kmeans = KMeans(n_clusters=num_clust, random_state=42, max_iter=1000, init='k-means++')
            kmeans.fit(df)
            inertia_list.append(kmeans.inertia_)
        kneedle = KneeLocator(
            number_clusters,
            inertia_list,
            curve='convex',
            direction='decreasing'
        )
        best_k = kneedle.elbow
        kmeans = KMeans(n_clusters=best_k, random_state=42, max_iter=1000, init='k-means++')
        kmeans.fit(df)
        labels = kmeans.predict(df)
        return self._convert_to_xarray(labels)


    def _convert_to_xarray(self, labels):
        labels = labels.reshape(self.data.shape[1], self.data.shape[2])
        segments = quickshift(
            labels,
            kernel_size=1,
            convert2lab=False,
            max_dist=2,
            ratio=1.0

        )
        labels = scipy.ndimage.median(
            input=labels,
            labels=segments,
            index=segments
        )

        labels = np.expand_dims(labels, axis=0)
        labels = xr.DataArray(
            data=labels,
            dims=['band', 'y', 'x'],
            coords={
                'band': ['labels'],
                'y': self.data.y.data,
                'x': self.data.x.data,
            },
            attrs={'long_name': ['labels']}
        )
        labels = labels.rio.write_crs(self.data.rio.crs)
        return labels

    def _check_transformer_input(self):
        """
        Args:

        Returns:
            None
        """
        if self.image_path is not None:
            if not os.path.exists(self.image_path):
                raise FileNotFoundError(f"Image not found at {self.image_path}")

        elif self.image_program not in ['Planet', 'Sentinel']:
            raise ValueError("Program must be either Planet or Sentinel.")
        self._check_program(self.image_program)


    def _read_image(self) -> xr.Dataset:
        """
        Reads and returns the image from the image_path.
        Args:
            None

        Returns:
            xr.DataArray: DataArray of the image read from the image_path.
        """
        return rioxarray.open_rasterio(self.image_path, engine='rasterio')



    def _generate_indexes(self):
        """
        Generates indexes from the image.
        Args:
            None

        Returns:
            None
        """
        index_transformer = IndexTransformer(program=self.image_program)
        return index_transformer.transform(self.data)


    def _generete_segments(self):
        """
        Generates segments from the image.
        Args:
            None

        Returns:
            None
        """
        segments_transformer = SegmentsTransformer()
        return segments_transformer.transform(self.data)




