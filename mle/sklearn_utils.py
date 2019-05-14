import numpy as np
import pandas as pd


def print_random_forest_feature_importance(trained_classifier, train_x: pd.DataFrame):
    importances = trained_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(train_x.shape[1]):
        print(f'{f + 1}. {train_x.columns[indices[f]]} {(importances[indices[f]])}')
