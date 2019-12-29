import pandas as pd


def load_train_csv(train_path):
    print('Loading training data...', end='')
    train_data = pd.read_csv(train_path)
    print('done!')
    return train_data
