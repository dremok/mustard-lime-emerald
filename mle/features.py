from typing import Union, List

import pandas as pd

Num = Union[int, float]


def transform_categorical(train_x: pd.DataFrame, test_x: pd.DataFrame, categorical_columns: List[str],
                          lowest_freq=0.005, one_hot_encode=True):
    """
    Transform categorical features in Pandas dataframes, consistently over train and test data.

    :param train_x: a Pandas dataframe containing the features of the train set
    :param test_x: a Pandas dataframe containing the features of the test set
    :param categorical_columns: columns containing the raw categorical feature data
    :param lowest_freq: values with less occurences than this frequency will be replace by 'other'
    :param one_hot_encode: if True, one-hot-encode the categorical columns and drop the original ones

    :return: a tuple (train_df, test_x) containing the transformed train and test sets
    """
    for col in categorical_columns:
        train_x[col] = train_x[col].fillna('missing').str.lower().str.strip().str.replace('[^a-z0-9 ]', '')
        test_x[col] = test_x[col].fillna('missing').str.lower().str.strip().str.replace('[^a-z0-9 ]', '')
        min_num_examples = int(train_x[col].count() * lowest_freq)
        to_keep = train_x[col].value_counts()[train_x[col].value_counts() >= min_num_examples].keys()
        to_keep = set(to_keep) & set(test_x[col])
        train_x.loc[~train_x[col].isin(to_keep), col] = 'other'
        test_x.loc[~test_x[col].isin(to_keep), col] = 'other'
        # Attention: Do not one-hot-encode for catboost
        if one_hot_encode:
            train_x = pd.concat([train_x, pd.get_dummies(train_x[col], prefix=col)], sort=False, axis=1) \
                .drop(col, axis=1)
            test_x = pd.concat([test_x, pd.get_dummies(test_x[col], prefix=col)], sort=False, axis=1) \
                .drop(col, axis=1)
    return train_x, test_x


def transform_numerical(train_x: pd.DataFrame, test_x: pd.DataFrame, numerical_columns: List[Num]):
    for col in numerical_columns:
        med = train_x[col].median()
        train_x[col].fillna(med, inplace=True)
        test_x[col].fillna(med, inplace=True)
    return train_x, test_x


def transform_sparse_to_boolean(train_x: pd.DataFrame, test_x: pd.DataFrame, to_boolean_columns: List):
    for col in to_boolean_columns:
        train_x[col] = train_x[col].notnull().astype('bool')
        train_x.rename(index=str, columns={col: f'has_{col}'}, inplace=True)
        test_x[col] = test_x[col].notnull().astype('bool')
        test_x.rename(index=str, columns={col: f'has_{col}'}, inplace=True)
    return train_x, test_x
