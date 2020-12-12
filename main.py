from sklearn.datasets import load_iris
from pprint import pprint
# from decision_tree import DecisionTreeClassifier
from dataset import Dataset

if __name__ == '__main__':
    # iris = load_iris()

    train_dataset = Dataset('ID3/monks/monks-1.train')
    test_dataset = Dataset('ID3/monks/monks-1.test')

    # y = iris.target

    # clf = DecisionTreeClassifier(max_depth=7, columns_names=iris.feature_names)
    # m = clf.fit(x, y)

    # pprint(m)

