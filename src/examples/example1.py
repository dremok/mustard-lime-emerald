import pandas as pd

from sklearn.linear_model import LinearRegression

from mle import engine
from mle import evaluate
from mle.problem_type import ProblemType


def main():
    model_config = {
        'problem_type': ProblemType.REGRESSION,
        'model_type': LinearRegression(),
        'feature_columns': [
            ('LotArea', int),
            ('Street', str),
            ('YrSold', pd.Timestamp),  # TODO: How to specify date format? DateTimeFormat?
            ('SaleCondition', str)
        ],
        'target_column': 'SalePrice',
        'evaluation': evaluate.CrossValidation()
    }
    engine.create_model(model_config, 'example_data/train.csv')


if __name__ == '__main__':
    main()
