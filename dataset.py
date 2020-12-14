import numpy as np


class Dataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.open_data()
        self.x = self.data[:, 1:]
        self.y = self.data[:, 0]

    def __len__(self):
        return len(self.data)

    def open_data(self):
        with open(self.data_dir, 'r') as f:
            data = f.readlines()

        data = [d.split(' ')[1:-1] for d in data]
        data = np.array(data, dtype=int)

        return data

    def __str__(self):
        return '{}'.format(self.data)

    def __call__(self, percent: float = 1.0):
        self.x = self.data[:, 1:]
        self.y = self.data[:, 0]
        data_population = int(self.__len__() * percent)
        choices = np.random.choice(self.__len__(), data_population, replace=False)
        self.x = self.x[choices]
        self.y = self.y[choices]
