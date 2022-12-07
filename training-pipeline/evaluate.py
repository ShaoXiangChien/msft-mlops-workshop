import pickle
import pandas as pd
from sklearn.metrics import precision_score
from datetime import datetime as dt
import smtplib, ssl
from azureml.core import Workspace, Model
import os

def load_model():
    with open("./model/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def load_data():
    return pd.read_csv("./X_test.csv"), pd.read_csv("./y_test.csv")

def register_model(score: float):
    print("registering the model")
    now = dt.now()
    ws = Workspace.from_config()

    model = Model.register(model_path="./model",
                            model_name="diabetes_model",
                            version=f"{now.year-1}{now.month}",
                            tags={'tags': "test", "date": now.strftime("%Y-%m-%d")},
                            description="svm model to predict diabetes",
                            workspace=ws)
    email_report(True, {
        "name": "diabetes-model",
        "version": f"{now.year}{now.month}",
        "tags": "test",
        "accuracy": f"{score * 100 :.2f}%"
    })

def email_report(better: bool, info: dict):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = os.getenv("senderEmail")  # Enter your address
    receiver_email = os.getenv("receiverEmail")  # Enter receiver address
    password = os.getenv("emailMFAToken")
    message = f"""
    The newly trained model has outperformed the previous one with accuracy of {info['accuracy']}.
    
    Model Info
    name: {info['name']},
    version: {info['version']},
    tag: {info['tags']} 

    registered to model registry
    """ if better else f"""
    The newly trained model has an accuracy of {info['accuracy']}, discarded...
    """

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

def main():
    print("load data and model")
    X_test, y_test = load_data()
    model = load_model()
    print("start inferencing")
    y_pred = model.predict(X_test)
    score = precision_score(y_test, y_pred)
    print(f"The precision of this model: {score}")
    register_model(score)
    # threshold = 0.75
    # if score > threshold:
    #     print("This model will be registered")
    #     register_model()
    # else:
    #     print("Accuracy lower than {threshold}, this model won't be registered")
    # compare with old model, with new test set


if __name__ == "__main__":
    main()
