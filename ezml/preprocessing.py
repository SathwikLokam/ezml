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
        self.encoder_classes = {}  # ⭐ NEW: fast lookup for unseen handling

        self.num_cols = None
        self.cat_cols = None

    # ================= FIT =================
    def fit_transform(self, X):
        X = X.copy()

        self.num_cols = X.select_dtypes(include=["number"]).columns
        self.cat_cols = X.select_dtypes(include=["object"]).columns

        # ---------- numeric ----------
        if len(self.num_cols) > 0:
            self.num_imputer = SimpleImputer(strategy="mean")
            X[self.num_cols] = self.num_imputer.fit_transform(X[self.num_cols])

        # ---------- categorical ----------
        if len(self.cat_cols) > 0:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            X[self.cat_cols] = self.cat_imputer.fit_transform(X[self.cat_cols])

            for col in self.cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

                self.encoders[col] = le
                self.encoder_classes[col] = set(le.classes_)  # ⭐ NEW

        # ---------- scaling ----------
        if self.scale and len(self.num_cols) > 0:
            self.scaler = StandardScaler()
            X[self.num_cols] = self.scaler.fit_transform(X[self.num_cols])

        return X

    # ================= TRANSFORM =================
    def transform(self, X):
        X = X.copy()
        # ---------- attempt numeric coercion (NEW) ----------
        for col in X.columns:
            if X[col].dtype == "object":
                converted = pd.to_numeric(X[col], errors="coerce")

                # if majority values are numeric → treat as numeric
                non_null_ratio = converted.notna().mean()

                if non_null_ratio > 0.9:  # heuristic threshold
                    X[col] = converted

        # ---------- numeric ----------
        if self.num_imputer is not None and len(self.num_cols) > 0:
            X[self.num_cols] = self.num_imputer.transform(X[self.num_cols])

        # ---------- categorical ----------
        if self.cat_imputer is not None and len(self.cat_cols) > 0:
            X[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])

            for col in self.cat_cols:
                if col not in self.encoders:
                    raise ValueError(f"Unexpected categorical column: {col}")

                le = self.encoders[col]
                known_classes = self.encoder_classes[col]

                # ⭐ PRODUCTION SAFE unseen handling
                X[col] = X[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known_classes else -1
                )

        # ---------- scaling ----------
        if self.scale and self.scaler is not None and len(self.num_cols) > 0:
            X[self.num_cols] = self.scaler.transform(X[self.num_cols])

        return X