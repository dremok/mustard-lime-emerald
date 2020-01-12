from sklearn.ensemble import RandomForestRegressor

import goal_metrics
from mle import engine
from mle.problem_type import ProblemType


def main():
    model_config = {
        'problem_type': ProblemType.REGRESSION,
        'model_class': RandomForestRegressor,
        'train_data_path': 'example_data/train.csv',
        'id_column': 'Id',
        'target_column': 'SalePrice',
        'goal_metric': goal_metrics.ROOT_MEAN_SQUARED_LOG_ERROR
    }
    engine.execute(model_config)


if __name__ == '__main__':
    main()
