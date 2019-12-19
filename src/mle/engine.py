from numbers import Number
from pandas.api.types import is_numeric_dtype
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale

from evaluate import Evaluator
from model_config import ModelConfig


def create_model(model_config_dict):
    model_config: ModelConfig = ModelConfig(model_config_dict)
    print(model_config)
    print()

    print('Loading training data...', end='')
    train_data = pd.read_csv(model_config.train_data_path)
    features_df = train_data[[x[0] for x in model_config.feature_column_tuples]].copy()
    feature_col_names = []
    for col_tup in model_config.feature_column_tuples:
        col_name = col_tup[0]
        col_transformer = col_tup[1]
        feature_col = features_df[col_name]
        if issubclass(col_transformer, Number):
            feature_col.astype(col_transformer)
        elif issubclass(col_transformer, TransformerMixin):
            features_df.loc[:, col_name] = col_transformer().fit_transform(feature_col)
        if model_config.standardize and is_numeric_dtype(features_df[col_name]):
            features_df.loc[:, col_name] = scale(features_df.loc[:, col_name])
        feature_col_names.append(col_name)
    X = features_df[feature_col_names].values
    y = train_data[model_config.target_column].values
    print('done!')

    print('Creating instance of model...', end='')
    model: BaseEstimator = model_config.model_type()
    pipeline = Pipeline([(model_config.model_type.__name__, model)])
    print('done!')
    print()

    evaluator: Evaluator = model_config.evaluator(pipeline)
    evaluator.evaluate(X, y)
    print()

    print('Training model on entire dataset...', end='')
    print('done!')
