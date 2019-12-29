from mle.problem_type import ProblemType

PROBLEM_TYPE = 'problem_type'
MODEL_CLASS = 'model_class'
TRAIN_DATA_PATH = 'train_data_path'
FEATURE_COLUMNS = 'feature_columns'
TARGET_COLUMN = 'target_column'
STANDARDIZE = 'standardize'
EVALUATOR = 'evaluator'


class ModelConfig:
    def __init__(self, model_config):
        self._problem_type: ProblemType = model_config[PROBLEM_TYPE]
        self._model_class = model_config[MODEL_CLASS]
        self._train_data_path = model_config[TRAIN_DATA_PATH]
        self._feature_columns = model_config[FEATURE_COLUMNS]
        self._target_column = model_config[TARGET_COLUMN]
        self._standardize = model_config[STANDARDIZE]

    @property
    def problem_type(self):
        return self._problem_type

    @property
    def model_class(self):
        return self._model_class

    @property
    def train_data_path(self):
        return self._train_data_path

    @property
    def feature_column_tuples(self):
        return self._feature_columns

    @property
    def target_column(self):
        return self._target_column

    @property
    def standardize(self):
        return self._standardize

    def __repr__(self):
        feature_col_str = ', '.join(
            [f'{x[0]} => {x[1].__name__}' if isinstance(x, tuple)
             else x
             for x in self._feature_columns]
        )
        t = self._target_column
        return 'Model Config:\n' + \
               '\n'.join([
                   f'problem_type = {self._problem_type.value}',
                   f'model_class = {self._model_class.__name__}',
                   f'train_data_path = {self._train_data_path}',
                   f'feature_columns = [{feature_col_str}]',
                   f'target_column = {(t[0], t[1].__name__) if isinstance(t, tuple) else t}',
               ])
