import data_loader
import evaluation
from model_config import ModelConfig
from pipeline import GeneralPipeline


def execute(model_config_dict):
    model_config: ModelConfig = ModelConfig(model_config_dict)
    print(model_config)
    print()

    train_df = data_loader.load_train_csv(model_config.train_data_path)
    pipeline: GeneralPipeline = GeneralPipeline.create(model_config, train_df)
    evaluation.evaluate(pipeline, train_df, model_config.goal_metric)

    print()
    print('Training model on entire dataset...', end='')
    print('done!')
