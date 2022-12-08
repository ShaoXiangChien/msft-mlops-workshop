from azureml.core import Workspace, Model
from datetime import datetime as dt



def upload_prod_model():
    ws = Workspace.from_config("./model")
    now = dt.now()
    ws.models['diabetes_model'].download("./", exist_ok=True)
    model = Model.register(model_path="model",
                            model_name="diabetes_model_prod",
                            tags={'tags': "prod", "date": now.strftime("%Y-%m-%d")},
                            description="svm model to predict diabetes",
                            workspace=ws)

if __name__ == "__main__":
    upload_prod_model()