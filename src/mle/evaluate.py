from abc import ABC

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, mean_squared_error, make_scorer
from sklearn.model_selection import cross_validate

from pipeline import GeneralPipeline

metrics_dict = {
    'mean_absolute_error': make_scorer(mean_squared_error),
    'root_mean_squared_error': make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)),
    'root_mean_squared_log_error': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true, y_pred)))
}


class Evaluator(ABC):
    def evaluate(self, pipeline: GeneralPipeline, df: pd.DataFrame):
        raise NotImplementedError


class CrossValidationEvaluator(Evaluator):
    def evaluate(self, pipeline: GeneralPipeline, train_df: pd.DataFrame, cv=5):
        print(f'Evaluating using cross-validation with {cv} folds...', end='')
        X = pipeline.prepare_X(train_df)
        y = pipeline.prepare_y(train_df)

        score_dict = cross_validate(pipeline.estimator, X, y, cv=cv, scoring=metrics_dict)
        print('done!')
        print()
        for metric in metrics_dict:
            scores = score_dict[f'test_{metric}']
            print(f'{metric}: {scores.mean()} (+/- {scores.std() * 2})')
