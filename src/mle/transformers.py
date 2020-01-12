from abc import ABC
from typing import List

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class BaseTransformer(ABC):
    def transform(self, df, col_name):
        raise NotImplementedError


class NumericTransformer(BaseTransformer):
    def __init__(self):
        self._pipeline: List[TransformerMixin] = [StandardScaler(), SimpleImputer()]

    def transform(self, df, col_name):
        transformed_col = df[[col_name]]
        for transformer in self._pipeline:
            transformed_col = transformer.fit_transform(pd.DataFrame(transformed_col))
        df.loc[:, col_name] = transformed_col
        return df


class CategoricalTransformer(BaseTransformer):
    def transform(self, df, col_name):
        df.loc[:, col_name] = SimpleImputer(strategy='most_frequent').fit_transform(df[[col_name]])
        df = pd.concat([df, pd.get_dummies(df[col_name], prefix=col_name)], axis=1)
        df.drop([col_name], axis=1, inplace=True)
        return df
