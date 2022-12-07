import json
import pandas as pd
import pickle
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model/model.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)



def run(data):
    test = json.loads(data)

    df = pd.DataFrame(test)
    X = df.drop(["PatientID"], axis=1)
    y_pred = model.predict(X)
    df['Prediction'] = y_pred
    print(f"received data {test}")
    return df.to_json()
