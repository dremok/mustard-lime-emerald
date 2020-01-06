from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import goal_metrics
from mle import engine
from mle.problem_type import ProblemType


def main():
    model_config = {
        'problem_type': ProblemType.REGRESSION,
        'model_class': RandomForestRegressor,
        'train_data_path': 'example_data/train.csv',
        'feature_columns': [
            ('LotArea', int),
            ('YearBuilt', int),
            'HouseStyle',
            ('YrSold', int),
            ('SaleCondition', LabelEncoder)
        ],
        'target_column': 'SalePrice',
        'standardize': True,
        'goal_metric': goal_metrics.ROOT_MEAN_SQUARED_LOG_ERROR
    }
    engine.execute(model_config)


if __name__ == '__main__':
    main()
