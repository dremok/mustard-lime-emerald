from pandas.api.types import is_numeric_dtype, is_string_dtype

from mle.problem_type import ProblemType
from transformers import NumericTransformer, CategoricalTransformer

PROBLEM_TYPE = 'problem_type'
MODEL_CLASS = 'model_class'
TRAIN_DATA_PATH = 'train_data_path'
ID_COLUMN = 'id_column'
FEATURE_COLUMNS = 'feature_columns'
TARGET_COLUMN = 'target_column'
GOAL_METRIC = 'goal_metric'


class ModelConfig:
    def __init__(self, model_config):
        self._problem_type: ProblemType = model_config[PROBLEM_TYPE]
        self._model_class = model_config[MODEL_CLASS]
        self._train_data_path = model_config[TRAIN_DATA_PATH]
        self._id_column = model_config.get(ID_COLUMN, None)
        self._feature_column_tuples = model_config.get(FEATURE_COLUMNS, None)
        self._target_column = model_config[TARGET_COLUMN]
        self._goal_metric = model_config[GOAL_METRIC]

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
    def id_column(self):
        return self._id_column

    @property
    def feature_column_tuples(self):
        return self._feature_column_tuples

    @property
    def target_column(self):
        return self._target_column

    @property
    def goal_metric(self):
        return self._goal_metric

    def __repr__(self):
        t = self._target_column
        return 'Model Config:\n' + \
               '\n'.join([
                   f'problem_type = {self.problem_type.value}',
                   f'model_class = {self.model_class.__name__}',
                   f'train_data_path = {self.train_data_path}',
                   f'feature_columns = [{self.feature_column_tuples}]',
                   f'target_column = {(t[0], t[1].__name__) if isinstance(t, tuple) else t}',
                   f'goal_metric = {self.goal_metric}',
               ])

    def set_defaults(self, train_df):
        if self.feature_column_tuples is None:
            self._set_default_features(train_df, self.problem_type)

    def _set_default_features(self, train_df, problem_type):
        features_df = train_df.copy().drop(self.target_column, axis=1)
        if self.id_column:
            features_df = features_df.drop(self.id_column, axis=1)
        if problem_type == ProblemType.REGRESSION:
            self._feature_column_tuples = [(x, NumericTransformer())
                                           for x in features_df.columns
                                           if is_numeric_dtype(features_df[x])]
            self._feature_column_tuples.extend(
                [(x, CategoricalTransformer())
                 for x in features_df.columns
                 if is_string_dtype(features_df[x])])
