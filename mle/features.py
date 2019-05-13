import pandas as pd


def transform_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_columns: [str],
                          lowest_freq=0.005, one_hot_encode=True):
    """
    Transform categorical features in Pandas dataframes, consistently over train and test data.

    :param train_df: a Pandas dataframe containing the features of the train set
    :param test_df: a Pandas dataframe containing the features of the test set
    :param categorical_columns: columns containing the raw categorical feature data
    :param lowest_freq: values with less occurences than this frequency will be replace by 'other'
    :param one_hot_encode: if True, one-hot-encode the categorical columns and drop the original ones

    :return: a tuple (train_df, test_df) containing the transformed train and test sets
    """
    for col in categorical_columns:
        train_df[col] = train_df[col].fillna('missing').str.lower().str.strip().str.replace('[^a-z0-9 ]', '')
        test_df[col] = test_df[col].fillna('missing').str.lower().str.strip().str.replace('[^a-z0-9 ]', '')
        min_num_examples = int(train_df[col].count() * lowest_freq)
        to_keep = train_df[col].value_counts()[train_df[col].value_counts() >= min_num_examples].keys()
        to_keep = set(to_keep) & set(test_df[col])
        train_df.loc[~train_df[col].isin(to_keep), col] = 'other'
        test_df.loc[~test_df[col].isin(to_keep), col] = 'other'
        # Attention: Do not one-hot-encode for catboost
        if one_hot_encode:
            train_df = pd.concat([train_df, pd.get_dummies(train_df[col], prefix=col)], sort=False, axis=1) \
                .drop(col, axis=1)
            test_df = pd.concat([test_df, pd.get_dummies(test_df[col], prefix=col)], sort=False, axis=1) \
                .drop(col, axis=1)
    return train_df, test_df
