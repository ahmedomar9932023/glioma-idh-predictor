from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureNameCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(getattr(X, "columns", [f"x{i}" for i in range(np.asarray(X).shape[1])]))
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

