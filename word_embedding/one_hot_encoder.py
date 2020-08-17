import pandas as pd


class Onehot:

    def __init__(self):
        self.vec_onehot = None

    def transform(self, data):
        df = pd.DataFrame({'item': data})
        df = pd.get_dummies(df)
        df.index = data
        df = df[~df.index.duplicated(keep='first')]
        self.vec_onehot = df
        return df

    def get_params(self, _index):
        return list(self.vec_onehot.loc[_index].to_dict().values())