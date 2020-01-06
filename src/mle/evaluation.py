import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, mean_squared_error, make_scorer
from sklearn.model_selection import cross_validate

from goal_metrics import ROOT_MEAN_SQUARED_ERROR, ROOT_MEAN_SQUARED_LOG_ERROR
from pipeline import GeneralPipeline

_metrics_dict = {
    ROOT_MEAN_SQUARED_ERROR: make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)),
    ROOT_MEAN_SQUARED_LOG_ERROR: make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true, y_pred)))
}


def evaluate(pipeline: GeneralPipeline, train_df: pd.DataFrame, goal_metric: str, cv=5):
    print(f'Evaluating using cross-validation with {cv} folds...', end='')
    X = pipeline.prepare_X(train_df)
    y = pipeline.prepare_y(train_df)

    scoring_dict = {goal_metric: _metrics_dict[goal_metric]}
    score_dict = cross_validate(pipeline.estimator, X, y, cv=cv, scoring=scoring_dict)
    print('done!')
    print()
    for metric in scoring_dict:
        scores = score_dict[f'test_{metric}']
        print(f'{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})')
