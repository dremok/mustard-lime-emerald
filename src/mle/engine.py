import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier

import data_loader
import evaluation
from mle.problem_type import ProblemType
from model_config import ModelConfig
from pipeline import GeneralPipeline


def execute(model_config_dict):
    model_config: ModelConfig = ModelConfig(model_config_dict)
    print(model_config)
    print()

    train_df = data_loader.load_train_csv(model_config.train_data_path)

    model_config.set_defaults(train_df)

    if model_config.problem_type == ProblemType.REGRESSION:
        for strategy in ['mean', 'median']:
            pipeline: GeneralPipeline = GeneralPipeline(model_config, DummyRegressor(strategy), train_df)
            evaluation.evaluate(pipeline, train_df, model_config.goal_metric)
    if model_config.problem_type == ProblemType.CLASSIFICATION:
        for strategy in ['stratified', 'most_frequent', 'uniform']:
            pipeline: GeneralPipeline = GeneralPipeline(model_config, DummyClassifier(strategy), train_df)
            evaluation.evaluate(pipeline, train_df, model_config.goal_metric)

    pipeline: GeneralPipeline = GeneralPipeline(model_config, model_config.model_class(), train_df)
    evaluation.evaluate(pipeline, train_df, model_config.goal_metric)

    df = pd.read_csv('./example_data/test.csv')
    pipeline.train_predict(train_df, df)
