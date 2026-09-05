from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TopPublishersEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, num):
        self.num = num
        self.top_pub = None

    def fit(self, X, y=None):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        else:
            series = pd.Series(X.ravel())
            
        self.top_pub = series.value_counts().iloc[:self.num].index.tolist()
        return self

    def transform(self, X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            col = X_copy.columns[0]
            mask = (~X_copy[col].isin(self.top_pub)) & (X_copy[col].notna())
            X_copy.loc[mask, col] = "Other"
            return X_copy
        else:
            series = pd.Series(X.ravel())
            mask = (~series.isin(self.top_pub)) & (series.notna())
            series.loc[mask] = "Other"
            return series.to_numpy().reshape(-1, 1)
    
    def get_feature_names_out(self, input_features=None):
        return input_features
