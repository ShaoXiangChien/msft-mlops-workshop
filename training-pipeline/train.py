from sklearn import svm
import pickle
import pandas as pd


def train(X_train, y_train):
    svm_model = svm.SVC()
    # fit the svm_model on the whole dataset
    svm_model.fit(X_train, y_train)

    with open("./trainging-pipeline/model/model.pkl", "wb") as fh:
        pickle.dump(svm_model, fh)

def load_data():
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    return X_train, y_train

if __name__ == "__main__":
    X_train, y_train = load_data()
    train(X_train, y_train)