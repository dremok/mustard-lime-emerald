import data_loader
from evaluate import CrossValidationEvaluator
from model_config import ModelConfig
from pipeline import GeneralPipeline


def train_model(model_config_dict):
    model_config: ModelConfig = ModelConfig(model_config_dict)
    print(model_config)
    print()

    train_df = data_loader.load_train_csv(model_config.train_data_path)
    pipeline: GeneralPipeline = GeneralPipeline.create(model_config, train_df)
    evaluator = CrossValidationEvaluator()
    evaluator.evaluate(pipeline, train_df)

    print()
    print('Training model on entire dataset...', end='')
    print('done!')
