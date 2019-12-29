from numbers import Number

import pandas as pd
from pandas.core.dtypes.common import is_integer_dtype, is_float_dtype, is_string_dtype, is_numeric_dtype
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale

from model_config import ModelConfig


def _default_transformer(col, train_df):
    if is_integer_dtype(train_df[col]):
        return int
    if is_float_dtype(train_df[col]):
        return float
    if is_string_dtype(train_df[col]):
        return LabelEncoder


class GeneralPipeline:
    def __init__(self, model_config: ModelConfig, train_df: pd.DataFrame):
        self._train_dtypes = train_df.dtypes
        self._feature_column_names = [x[0] if isinstance(x, tuple) else x for x in model_config.feature_column_tuples]
        self._feature_column_transformers = [
            x[1] if isinstance(x, tuple)
            else _default_transformer(x, train_df)
            for x in model_config.feature_column_tuples
        ]
        if isinstance(model_config.target_column, tuple):
            self._target_column_name = model_config.target_column[0]
            self._target_column_transformer = model_config.target_column[1]
        else:
            self._target_column_name = model_config.target_column
            self._target_column_transformer = _default_transformer(self._target_column_name, train_df)
        self._standardize = model_config.standardize
        self._model: BaseEstimator = model_config.model_class()
        self._estimator = Pipeline([(model_config.model_class.__name__, self._model)])

    @classmethod
    def create(cls, model_config: ModelConfig, train_df: pd.DataFrame):
        return GeneralPipeline(model_config, train_df)

    @property
    def estimator(self):
        return self._estimator

    def prepare_X(self, df):
        # TODO: Validate that dtypes is same as in training
        features_df = df[self._feature_column_names].copy()
        for col_name, col_transformer in zip(self._feature_column_names, self._feature_column_transformers):
            feature_col = features_df[col_name]
            if issubclass(col_transformer, Number):
                feature_col.astype(col_transformer)
            elif issubclass(col_transformer, TransformerMixin):
                features_df.loc[:, col_name] = col_transformer().fit_transform(feature_col)
            if self._standardize and is_numeric_dtype(features_df[col_name]):
                features_df.loc[:, col_name] = scale(features_df.loc[:, col_name])
        X = features_df[self._feature_column_names].values
        return X

    def prepare_y(self, df):
        target_col = df[[self._target_column_name]]
        if issubclass(self._target_column_transformer, Number):
            target_col.astype(self._target_column_transformer)
        elif issubclass(self._target_column_transformer, TransformerMixin):
            df.loc[:, self._target_column_transformer] = self._target_column_transformer().fit_transform(target_col)
        y = target_col.values.ravel()
        return y
