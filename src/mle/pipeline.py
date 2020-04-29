from numbers import Number
from typing import List

import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype, is_string_dtype, is_numeric_dtype
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

from model_config import ModelConfig
from transformers import BaseTransformer


# TODO: Merge with set default feature names
def _default_transformer(col, train_df):
    if is_integer_dtype(train_df[col]):
        return int
    if is_float_dtype(train_df[col]):
        return float
    if is_string_dtype(train_df[col]):
        return LabelEncoder


class GeneralPipeline:
    def __init__(self, model_config: ModelConfig, predictor, train_df: pd.DataFrame):
        self._train_dtypes = train_df.dtypes
        self._id_column = model_config.id_column
        self._feature_column_names = [x[0] if isinstance(x, tuple) else x for x in model_config.feature_column_tuples]
        self._feature_column_transformers: List[BaseTransformer] = [
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
        self._predictor = predictor

    @property
    def predictor(self):
        return self._predictor

    def prepare_X(self, main_df, other_df=None):
        features_df = self._transform_features(main_df)
        if other_df is not None:
            features_df = self._remove_features_not_in_test(features_df, other_df)
        X = features_df.values
        return X

    def prepare_y(self, df):
        target_col = df[[self._target_column_name]]
        if issubclass(self._target_column_transformer, Number):
            target_col.astype(self._target_column_transformer)
        elif issubclass(self._target_column_transformer, TransformerMixin):
            df.loc[:, self._target_column_transformer] = self._target_column_transformer().fit_transform(target_col)
        y = target_col.values.ravel()
        return y

    def _transform_features(self, df):
        df = df[self._feature_column_names].copy()
        cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
        for col in cols_with_missing:
            df[f'{col}_missing'] = df[col].isnull()
        numeric_cols = [col for col in self._feature_column_names if is_numeric_dtype(df[col])]
        df.loc[:, numeric_cols] = KNNImputer().fit_transform(df[numeric_cols])
        for col_name, col_transformer in zip(self._feature_column_names, self._feature_column_transformers):
            df = col_transformer.transform(df, col_name)
        return df

    def _remove_features_not_in_test(self, features_df, test_df):
        test_df = self._transform_features(test_df)
        features_df = features_df.drop([x for x in features_df.columns if x not in test_df.columns], axis=1)
        return features_df

    def fit(self, train_df, test_df=None):
        X = self.prepare_X(train_df, test_df)
        y = self.prepare_y(train_df)
        self.predictor.fit(X, y)

    def predict(self, test_df: pd.DataFrame, train_df: pd.DataFrame):
        X = self.prepare_X(test_df, train_df)
        predictions = pd.DataFrame(self.predictor.predict(X))
        if self._id_column:
            predictions = pd.concat([test_df[[self._id_column]], predictions], axis=1)
        predictions.columns = [self._id_column, self._target_column_name]
        return predictions

    def train_predict(self, train_df, test_df):
        print()
        print('Training model on entire dataset...', end='')
        self.fit(train_df, test_df)
        print('done!')
        print('Predicting on test set...', end='')
        predictions = self.predict(test_df, train_df)
        print('done!')
        predictions.to_csv('submission.csv', index=False)
