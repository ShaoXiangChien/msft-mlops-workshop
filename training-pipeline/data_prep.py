from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pandas as pd
from sklearn.model_selection import train_test_split
import os


default_credential = DefaultAzureCredential()
# prepare two versions of data
blob = BlobClient(account_url=os.getenv("accountUrl"),
              container_name=os.getenv("containerName"),
              blob_name="diabetes-data",
              credential=os.getenv("blobConnCredential"))

def load_data():
    globals()
    with open("tmp.csv", "wb") as f:
        data = blob.download_blob()
        data.readinto(f)
    
    return pd.read_csv("tmp.csv")

def normalize_data(df):
    standardized_columns = [
    "PlasmaGlucose",
    "DiastolicBloodPressure",
    "TricepsThickness",
    "SerumInsulin",
    "BMI",
    "DiabetesPedigree",
    ]

    for col in standardized_columns:
        data = df[col]
        df[col] = df[col].apply(lambda x:( x - data.mean()) / data.std())
    
    return df

def split_n_store(df):
    label_col = "Diabetic"
    X = df.drop([label_col, "PatientID"], axis=1)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

def main():
    print("Loading data")
    df = load_data()
    print("data loaded")
    df = normalize_data(df)
    split_n_store(df)
    print("finish preprocessing and spliting")



if __name__ == "__main__":
    main()
