import math
import numpy as np


def entropy_func(_class, n):  # Entropy for one class
    return -(_class * 1.0 / n) * math.log(_class * 1.0 / n, 2)


def entropy_cal(_class1, _class2):  # Entropy for two class
    if _class1 == 0 or _class2 == 0:
        return 0
    return entropy_func(_class1, _class1 + _class2) + entropy_func(_class2, _class1 + _class2)


def entropy_of_one_division(division: np.ndarray):  # Entropy for multiple class
    s = 0
    n = len(division)
    classes = set(division)
    for _class in classes:
        n_c = sum(division == _class)
        e = (n_c * 1.0 / n) * entropy_cal(sum(division == _class), sum(division != _class))
        s += e

    return s, n


def get_entropy(y_predict, y_real):
    if len(y_predict) != len(y_real):
        print("They have to be the same length")
        return None

    n = len(y_real)
    s_true, n_true = entropy_of_one_division(y_real[y_predict])
    s_false, n_false = entropy_of_one_division(y_real[~y_predict])
    s = n_true * 1.0 / n * s_true + n_false * 1.0 / n * s_false
    return s
