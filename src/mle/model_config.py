from mle.problem_type import ProblemType

PROBLEM_TYPE = 'problem_type'
MODEL_TYPE = 'model_type'
FEATURE_COLUMNS = 'feature_columns'
TARGET_COLUMN = 'target_column'
EVALUATION = 'evaluation'


class ModelConfig:
    def __init__(self, model_config):
        self._problem_type: ProblemType = model_config[PROBLEM_TYPE]
        self._model_type = model_config[MODEL_TYPE]
        self._feature_columns = model_config[FEATURE_COLUMNS]
        self._target_column = model_config[TARGET_COLUMN]
        self._evaluation = model_config[EVALUATION]

    def __repr__(self):
        return 'Model Config:\n' + \
               '\n'.join([
                   f'problem_type = {self._problem_type.value}',
                   f'model_type = {self._model_type.__class__.__name__}',
                   f'feature_columns = [{", ".join([f"{x[0]} => {x[1].__name__}" for x in self._feature_columns])}]',
                   f'target_column = {self._target_column}',
               ])
