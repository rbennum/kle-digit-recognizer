import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib

def get_train_dataset(normalize=False) -> tuple[list, list]:
    df = pd.read_csv('train.csv')
    labels = df.iloc[:, 0]
    data = df.iloc[:, 1:]
    if normalize:
        data = min_max_normalization(data)
    data = data.to_numpy().tolist()
    return (labels, data)

def get_test_dataset(normalize=False) -> list:
    df = pd.read_csv('test.csv')
    if normalize:
        df = min_max_normalization(df)
    data = df.to_numpy().tolist()
    return data

def get_classifier() -> MLPClassifier:
    clf = MLPClassifier(
        hidden_layer_sizes=(100, 75, 50,),
        solver='sgd',
        verbose=True,
        max_iter=500
    )
    return clf

def train(normalize=False) -> MLPClassifier:
    labels, data = get_train_dataset(normalize=normalize)
    clf = get_classifier()
    clf.fit(data, labels)
    export_model(clf)
    return clf

def test(clf: MLPClassifier, normalize=False) -> list:
    data = get_test_dataset(normalize=normalize)
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

def min_max_normalization(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = result.astype('float32')
    for row in range(len(df)):
        result.iloc[row] = (
            (result.iloc[row] - result.iloc[row].min()) /
            (result.iloc[row].max() - result.iloc[row].min())
        )
    return result

if __name__ == '__main__':
    clf = train(normalize=True)
    result = test(clf, normalize=True)
    create_submission(result)