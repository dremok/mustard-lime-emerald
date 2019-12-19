from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from mle import engine
from mle import evaluate
from mle.problem_type import ProblemType


def main():
    model_config = {
        'problem_type': ProblemType.REGRESSION,
        'model_type': RandomForestRegressor,
        'train_data_path': 'example_data/train.csv',
        'feature_column_tuples': [
            ('LotArea', int),
            ('YearBuilt', int),
            'HouseStyle',
            ('YrSold', int),  # TODO: How to specify date format? DateTimeFormat?
            ('SaleCondition', LabelEncoder)
        ],
        'target_column': 'SalePrice',
        'standardize': True,
        'evaluator': evaluate.CrossValidation
    }
    engine.create_model(model_config)


if __name__ == '__main__':
    main()
