from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier
from dataset import Dataset
from tqdm import tqdm

if __name__ == '__main__':

    feature_name = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    train_dataset = Dataset('ID3/monks/monks-1.train')
    test_dataset = Dataset('ID3/monks/monks-1.test')

    round_acc = []
    for i in tqdm(range(500)):
        accuracies = []
        for data_percent in range(0, 100, 10):
            clf = DecisionTreeClassifier(max_depth=7, columns_names=feature_name)
            data_percent += 10
            train_dataset(data_percent / 100)
            x_train = train_dataset.x
            y_train = train_dataset.y

            x_test = test_dataset.x
            y_test = test_dataset.y

            m = clf.fit(x_train, y_train)
            predict = clf.predict(x_test)

            accuracy = sum(predict == y_test) / len(y_test)
            # print(accuracy)
            accuracies.append(accuracy)
        round_acc.append(accuracies)

    round_acc = np.array(round_acc)
    round_acc_mean = np.mean(round_acc, axis=0)
    plt.plot(list(range(10, 110, 10)), round_acc_mean, '*-')
    plt.grid(True)
    plt.show()
