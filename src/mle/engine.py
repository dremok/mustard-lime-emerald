from numbers import Number

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
    features_df = train_data[[x[0] for x in model_config.feature_column_tuples]]
    feature_col_names = []
    for col_tup in model_config.feature_column_tuples:
        col_name = col_tup[0]
        col_transformer = col_tup[1]
        feature_col = features_df[col_name]
        if issubclass(col_transformer, Number):
            feature_col.astype(col_transformer)
        elif issubclass(col_transformer, TransformerMixin):
            features_df.loc[:, col_name] = col_transformer().fit_transform(feature_col)
        feature_col_names.append(col_name)
    X = features_df[feature_col_names].values
    y = train_data[model_config.target_column].values
    if model_config.standardize:
        X = scale(X)
    print('done!')

    print('Creating instance of _pipeline...', end='')
    model: BaseEstimator = model_config.model_type()
    print('done!')
    print('Training _pipeline...', end='')
    pipeline = Pipeline([(model_config.model_type.__name__, model)])
    print('done!')

    evaluator: Evaluator = model_config.evaluator(pipeline)
    evaluator.evaluate(X, y)
