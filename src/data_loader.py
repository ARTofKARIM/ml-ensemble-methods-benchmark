"""Data loading for ensemble benchmark."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml

class DataLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load(self, path=None):
        path = path or self.config["data"]["path"]
        df = pd.read_csv(path)
        print(f"Loaded: {df.shape}")
        return df

    def preprocess(self, df):
        df = df.copy()
        for col in df.select_dtypes(include=["object"]).columns:
            if col != self.config["data"]["target"]:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        df = df.fillna(df.median(numeric_only=True))
        return df

    def split(self, df):
        target = self.config["data"]["target"]
        X = df.drop(columns=[target])
        y = df[target]
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"], stratify=y)
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X.columns)
        return X_train, X_test, y_train, y_test
