import pandas as pd
import numpy as np
import copy
from sklearn.utils import shuffle
import math


class MData:
    def __init__(self):
        self.train_data = None
        self.train_lable = None
        self.test_data = None
        self.test_lable = None
        self.num_of_batch = 1
        self.got_batch = 0

    def getdata(self, path, ratio=0.1, lable=-1):
        df = pd.read_csv(path, header=None)
        df = shuffle(df)
        df = df.reset_index(drop=True)

        r = int(df.shape[0]*(1 - ratio))

        train_set = df[:r].copy()
        col = copy.copy(list(df.columns))
        lb = col.pop(lable)
        self.train_data = train_set[col]
        test_set = df[r:].copy()
        self.test_data = test_set[col]

        num_of_class = len(df[lb].unique())
        tags = np.zeros((df.shape[0], num_of_class))
        for i in range(len(df[lb])):
            tags[i][df[lb][i]] = 1
        df_tags = pd.DataFrame(tags)
        self.train_lable = df_tags[:r]
        self.test_lable = df_tags[r:]
        return 0

    def get_batch(self, size):
        if self.got_batch == self.num_of_batch:
            rand_index = np.random.permutation(self.train_data.index)
            self.train_lable = self.train_lable.reindex(rand_index)
            self.train_data = self.train_data.reindex(rand_index)
            self.num_of_batch = math.floor(len(self.train_data)/size)
            self.got_batch = 0
        train_data = self.train_data[self.got_batch*size:(self.got_batch+1)*size]
        train_tag = self.train_lable[self.got_batch*size:(self.got_batch+1)*size]
        self.got_batch += 1
        return train_data, train_tag


if __name__ == '__main__':
    # d = MData()
    # d.getdata('./scale1.data.txt', 0.2)
    # print(d.train_data)
    # print(d.get_batch(1))
    df = pd.read_csv('./iris.data.txt', header=None)
    # df[5] = df[0]
    df[4] = df[4].map(lambda x: 0 if x == 'Iris-setosa' else x)
    df[4] = df[4].map(lambda x: 1 if x == 'Iris-versicolor' else x)
    df[4] = df[4].map(lambda x: 2 if x == 'Iris-virginica' else x)
    # df[5] = df[5].map(lambda x: 0 if x == 'R' else x)
    # df[5] = df[5].map(lambda x: 1 if x == 'B' else x)
    # df[5] = df[5].map(lambda x: 2 if x == 'L' else x)
    # df = df[[1,2,3,4,5]]
    print(df)
    df.to_csv('./iris1.data.txt', index=False)
