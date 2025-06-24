import pandas as pd
from sklearn.neural_network import MLPClassifier

def get_train_dataset() -> tuple[list, list]:
    df = pd.read_csv('train.csv')
    labels = df.iloc[:, 0]
    data = df.iloc[:, 1:].to_numpy().tolist()
    return (labels, data)

def get_test_dataset() -> tuple[list, list]:
    df = pd.read_csv('test.csv')
    labels = df.iloc[:, 0]
    data = df.iloc[:, 1:].to_numpy().tolist()
    return (labels, data)

def get_classifier() -> MLPClassifier:
    clf = MLPClassifier(
        hidden_layer_sizes=(50,),
        solver='sgd',
        verbose=True
    )
    return clf

def train() -> MLPClassifier:
    labels, data = get_train_dataset()
    clf = get_classifier()
    clf.fit(data, labels)
    return clf

def test(clf: MLPClassifier) -> list:
    _, data = get_test_dataset()
    result = clf.predict(data)
    return result

if __name__ == '__main__':
    labels, data = get_train_dataset()
    print(data[:5])