from entropy import get_entropy
import numpy as np


class DecisionTreeClassifier(object):
    def __init__(self, max_depth, columns_names):
        self.depth = 0
        self.max_depth = max_depth
        self.columns_name = columns_names

    def find_best_split(self, col, y):
        min_entropy = float('inf')
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = get_entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff

    def find_best_split_of_all(self, x, y):
        col = None
        min_entropy = float('inf')
        cutoff = None

        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y)
            if entropy == 0:
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy

    def fit(self, x, y, par_node={}, depth=0):
        if par_node is None:
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val': y[0]}
        elif depth >= self.max_depth:
            return None
        else:
            col, cutoff, entropy = self.find_best_split_of_all(x, y)
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]
            par_node = {'col': self.columns_name[col], 'index_col': col, 'cutoff': cutoff, 'val': np.round(np.mean(y))}
            par_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, {}, depth + 1)
            par_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, {}, depth + 1)

            self.depth += 1
            self.trees = par_node
            return par_node

    def all_same(self, items):
        return all(x == items[0] for x in items)

    def predict(self, x: np.ndarray):
        results = np.array([0]*len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results

    def _get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer:
            if cur_layer.get('cutoff'):
                if row[cur_layer['index_col']] < cur_layer['cutoff']:
                    cur_layer = cur_layer['left']
                else:
                    cur_layer = cur_layer['right']
            else:
                return cur_layer.get('val')
        else:
            return -1
