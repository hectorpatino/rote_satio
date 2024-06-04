import os

import numpy as np
import pandas as pd
import scipy
import xarray as xr
import rioxarray

from skimage.segmentation import quickshift

from rote_satio.models.segmentatition.planet.dispatcher import PlanetPipeline
from rote_satio.utils.geobia_transformer import SegmentsTransformer
from rote_satio.utils.index_transformer import IndexTransformer
from rote_satio.utils.utils import parse_to_pandas


class AutoSegmentation():
    def __init__(
            self,
            image_path: str,
            image_program: str,
            generate_objects: bool = True,
    ):
        """
        Args:
            image_path: Path to the image that needs to be segmented.
            image_program: Program of the sensor of the image.
            generate_objects: If True, there will be generation of geobias objects.

        Returns:
            AutoSegmentation: And AutoSegmentation object with the given image_path and image_program
            that can be used to perform segmentation on the image.
        """
        self._check_input(image_path, image_program)
        self.image_path = image_path
        self.image_program = image_program
        self.generate_objects = generate_objects
        self.umap_pipeline = None
        self.hdbscan_pipeline = None


    def predict(self):

        self._read_image()
        print('generating indexes')
        self._generate_indexes()
        if self.generate_objects:
            print('generating segments')
            self._generete_segments()
        df = parse_to_pandas(self.image)
        pd.DataFrame(data=df.columns).to_csv('columns_2.csv')
        pipeline = PlanetPipeline()
        print('fitting pipeline')
        labels = pipeline.predict(df)
        print('converting to xarray')
        return self._convert_to_xarray(labels)



    def _convert_to_xarray(self, labels):
        labels = labels.reshape(self.image.shape[1], self.image.shape[2])
        segments = quickshift(
            labels,
            kernel_size=1,
            convert2lab=False,
            max_dist=1,
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
                'y': self.image.y.data,
                'x': self.image.x.data,
            },
            attrs={'long_name': ['labels']}
        )
        labels = labels.rio.write_crs(self.image.rio.crs)
        return labels

    def _check_input(self, image_path: str, image_program: str):
        """
        Args:
            image_path: Path to the image that needs to be segmented.
            image_program: Program of the sensor of the image.

        Returns:
            None
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        if image_program not in ["Planet"]:
            raise ValueError(f"Invalid image program {image_program}. Supported programs are ['Planet']")

        if image_program is None:
            raise ValueError("Image program cannot be None")


    def _read_image(self) -> xr.Dataset:
        """
        Reads and returns the image from the image_path.
        Args:
            None

        Returns:
            xr.DataArray: DataArray of the image read from the image_path.
        """
        self.image = rioxarray.open_rasterio(self.image_path, engine='rasterio')



    def _generate_indexes(self):
        """
        Generates indexes from the image.
        Args:
            None

        Returns:
            None
        """
        index_transformer = IndexTransformer(program=self.image_program)
        self.image = index_transformer.transform(self.image)


    def _generete_segments(self):
        """
        Generates segments from the image.
        Args:
            None

        Returns:
            None
        """
        segments_transformer = SegmentsTransformer()
        self.image = segments_transformer.transform(self.image)




