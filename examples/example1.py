import pandas as pd
from sklearn import linear_model

import mle
from mle import ProblemType


def main():
    train_data = pd.read_csv('./example_data/train.csv')
    model_config = {
        'problem_type': ProblemType.REGRESSION,
        'model_type': linear_model.LinearRegression(),
        'feature_columns': [
            'LotFrontage',
            'LotArea',
            'Street',
            'Alley',
            'LotShape',
            'LandContour',
            'Utilities',
            'EnclosedPorch',
            '3SsnPorch',
            'ScreenPorch',
            'PoolArea',
            'PoolQC',
            'Fence',
            'MoSold',
            'YrSold',
            'SaleType',
            'SaleCondition'
        ],
        'target_column': 'SalePrice',
        'evaluation': mle.evaluate.CrossValidation()
    }
    mle.create_model(model_config, train_data)


if __name__ == '__main__':
    main()
