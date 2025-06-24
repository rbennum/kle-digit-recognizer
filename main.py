import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib

def get_train_dataset() -> tuple[list, list]:
    df = pd.read_csv('train.csv')
    labels = df.iloc[:, 0]
    data = df.iloc[:, 1:].to_numpy().tolist()
    return (labels, data)

def get_test_dataset() -> list:
    df = pd.read_csv('test.csv')
    data = df.to_numpy().tolist()
    return data

def get_classifier() -> MLPClassifier:
    clf = MLPClassifier(
        hidden_layer_sizes=(100, 75, 50,),
        solver='sgd',
        verbose=True
    )
    return clf

def train() -> MLPClassifier:
    labels, data = get_train_dataset()
    clf = get_classifier()
    clf.fit(data, labels)
    export_model(clf)
    return clf

def test(clf: MLPClassifier) -> list:
    data = get_test_dataset()
    result = clf.predict(data)
    return result

def export_model(clf: MLPClassifier):
    import time
    timestamp = int(time.time())
    joblib.dump(clf, f'models/clf_{timestamp}.pkl')

def import_model(name: str) -> MLPClassifier:
    model = joblib.load(name)
    return model

def create_submission(result: list):
    import time
    timestamp = int(time.time())
    df = pd.DataFrame({
        "ImageId": range(1, 28001),
        "Label": result
    })
    filename = f'submissions/submission_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f'created new submission: {filename}')

if __name__ == '__main__':
    clf = train()
    result = test(clf)
    create_submission(result)