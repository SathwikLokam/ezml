import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class Preprocessor:
    """Handles preprocessing for ezml."""

    def __init__(self, scale=False):
        self.scale = scale

        self.num_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.encoders = {}

        self.num_cols = None
        self.cat_cols = None

    # ================= FIT =================
    def fit_transform(self, X):
        X = X.copy()

        self.num_cols = X.select_dtypes(include=["number"]).columns
        self.cat_cols = X.select_dtypes(include=["object"]).columns

        # numeric
        if len(self.num_cols) > 0:
            self.num_imputer = SimpleImputer(strategy="mean")
            X[self.num_cols] = self.num_imputer.fit_transform(X[self.num_cols])

        # categorical
        if len(self.cat_cols) > 0:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            X[self.cat_cols] = self.cat_imputer.fit_transform(X[self.cat_cols])

            for col in self.cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le

        # scaling
        if self.scale and len(self.num_cols) > 0:
            self.scaler = StandardScaler()
            X[self.num_cols] = self.scaler.fit_transform(X[self.num_cols])

        return X

    # ================= TRANSFORM =================
    def transform(self, X):
        X = X.copy()

        # numeric
        if self.num_imputer is not None and len(self.num_cols) > 0:
            X[self.num_cols] = self.num_imputer.transform(X[self.num_cols])

        # categorical
        if self.cat_imputer is not None and len(self.cat_cols) > 0:
            X[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])

            for col in self.cat_cols:
                if col not in self.encoders:
                    raise ValueError(f"Unexpected categorical column: {col}")
                le = self.encoders[col]
                X[col] = le.transform(X[col].astype(str))

        # scaling
        if self.scale and self.scaler is not None and len(self.num_cols) > 0:
            X[self.num_cols] = self.scaler.transform(X[self.num_cols])

        return X