import os

from pathlib import Path
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.base import ClusterMixin
from sklearn.cluster import HDBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from umap import UMAP

from rote_satio.models.utils.utils import save_pipeline, load_pipeline


class PlanetPipeline(ClusterMixin):
    preprocessing_pipeline_name = 'preprocessing_pipeline.joblib'
    model_name = 'clustering_pipeline.joblib'
    current_file = Path(os.path.dirname(__file__))

    def _preprocessing_pipeline(self):
        return Pipeline([
            ('robust_scaler', SklearnTransformerWrapper(transformer=RobustScaler())),
            ('scaler', SklearnTransformerWrapper(transformer=MinMaxScaler())),
            ('discretizer', EqualWidthDiscretiser(bins=10)),
            ('scaler_2', SklearnTransformerWrapper(transformer=MinMaxScaler())),
            ('umap', UMAP(n_components=7, n_jobs=-1, n_neighbors=12)),
            ]
        )

    def _clustering_pipeline(self):
        return HDBSCAN(
            min_cluster_size=10,
            min_samples=50,
        )

    def fit_predict(self, X, y=None, **kwargs):
        preprocessing_pipeline = self._preprocessing_pipeline()
        preprocessed = preprocessing_pipeline.fit_transform(X)
        clustered = self._clustering_pipeline()
        return clustered.fit_predict(preprocessed)
