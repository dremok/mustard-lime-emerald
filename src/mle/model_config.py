from typing import Callable

from mle.problem_type import ProblemType

PROBLEM_TYPE = 'problem_type'
MODEL_TYPE = 'model_type'
TRAIN_DATA_PATH = 'train_data_path'
FEATURE_COLUMNS = 'feature_column_tuples'
TARGET_COLUMN = 'target_column'
STANDARDIZE = 'standardize'
EVALUATOR = 'evaluator'


class ModelConfig:
    def __init__(self, model_config):
        self._problem_type: ProblemType = model_config[PROBLEM_TYPE]
        self._model_type = model_config[MODEL_TYPE]
        self._train_data_path = model_config[TRAIN_DATA_PATH]
        self._feature_column_tuples = model_config[FEATURE_COLUMNS]
        self._target_column = model_config[TARGET_COLUMN]
        self._standardize = model_config[STANDARDIZE]
        self._evaluator = model_config[EVALUATOR]

    @property
    def problem_type(self):
        return self._problem_type

    @property
    def model_type(self):
        return self._model_type

    @property
    def train_data_path(self):
        return self._train_data_path

    @property
    def feature_column_tuples(self):
        return self._feature_column_tuples

    @property
    def target_column(self):
        return self._target_column

    @property
    def standardize(self):
        return self._standardize

    @property
    def evaluator(self) -> Callable:
        return self._evaluator

    def __repr__(self):
        return 'Model Config:\n' + \
               '\n'.join([
                   f'problem_type = {self._problem_type.value}',
                   f'model_type = {self._model_type.__name__}',
                   f'train_data_path = {self._train_data_path}',
                   f'feature_columns = [{", ".join([f"{x[0]} => {x[1].__name__}" for x in self._feature_column_tuples])}]',
                   f'target_column = {self._target_column}',
               ])
