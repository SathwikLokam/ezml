import os
import pandas as pd
from sklearn.model_selection import train_test_split

from .detection import detect_task
from .models import get_model
from .preprocessing import Preprocessor
from .persistence import save_object, load_object


class AutoModel:
    """
    Beginner-friendly AutoML for tabular data.
    """

    VERSION = "1.5"

    # ================= INIT =================
    def __init__(
        self,
        task="auto",
        mode="fast",
        preprocess=True,
        scale=False,
        verbose=True,
        random_state=42,
    ):
        self.task = task
        self.mode = mode
        self.use_preprocess = preprocess
        self.scale = scale
        self.verbose = verbose
        self.random_state = random_state

        self.model = None
        self.columns = None
        self.trained = False
        self.preprocessor = None

    # ================= TRAIN =================
    def train(self, file_path, target):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = pd.read_csv(file_path)

        if data.empty:
            raise ValueError("Dataset is empty.")

        data.columns = data.columns.str.strip()

        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found.")

        X = data.drop(columns=[target])
        y = data[target]

        if X.shape[1] == 0:
            raise ValueError("No feature columns found.")

        self.columns = X.columns.tolist()

        # ===== task detection =====
        if self.task == "auto":
            self.task = detect_task(y)
            if self.verbose:
                print("ℹAuto task detection used")

        if self.verbose:
            print(f"Detected task: {self.task}")

        # ===== preprocessing =====
        if self.use_preprocess:
            if self.verbose:
                print("Preprocessing data...")
            self.preprocessor = Preprocessor(scale=self.scale)
            X = self.preprocessor.fit_transform(X)

        # ===== model =====
        self.model = get_model(
            task=self.task,
            mode=self.mode,
            random_state=self.random_state,
        )

        if self.model is None:
            raise RuntimeError("Model creation failed. Check mode and dependencies.")

        if self.verbose:
            model_name = type(self.model).__name__
            print(f"Training {model_name}...")

        # ===== train =====
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)

        self.trained = True

        if self.verbose:
            metric = "Accuracy" if self.task == "classification" else "R2"
            print(f"Model trained | {metric}: {score:.4f}")

    # ================= PREDICT =================
    def predict(self, new_data):
        """
        Predict on new data.

        Supported input formats:
        - list: [v1, v2, v3]
        - list of lists: [[...], [...]]
        - dict: {"col": value}
        - list of dicts: [{"col": value}, ...]
        """

        import pandas as pd

        if not self.trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")

    # ---------------- NORMALIZE INPUT ----------------

    # Case 1: single dict
        if isinstance(new_data, dict):
            df = pd.DataFrame([new_data])

    # Case 2: list of dicts
        elif isinstance(new_data, list) and len(new_data) > 0 and isinstance(new_data[0], dict):
            df = pd.DataFrame(new_data)

    # Case 3: list or list of lists (old behavior)
        else:
            if not isinstance(new_data, list):
                raise ValueError("Unsupported input type for prediction.")

        # single row like [1,2,3]
            if len(new_data) > 0 and not isinstance(new_data[0], (list, tuple)):
                new_data = [new_data]

            if len(new_data[0]) != len(self.columns):
                raise ValueError(
                    f"Expected {len(self.columns)} features, got {len(new_data[0])}"
                )

            df = pd.DataFrame(new_data, columns=self.columns)

    # ---------------- COLUMN ALIGNMENT ----------------

        missing_cols = set(self.columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns for prediction: {missing_cols}")

    # ensure correct order
        df = df[self.columns]

    # ---------------- PREPROCESS ----------------

        if self.use_preprocess and self.preprocessor is not None:
            df = self.preprocessor.transform(df)

        return self.model.predict(df)
        


    # ================= SAVE =================
    def save(self, path):
        """Save full AutoModel pipeline."""
        if not self.trained:
            raise RuntimeError("Train the model before saving.")

        package = {
            "version": self.VERSION,
            "task": self.task,
            "columns": self.columns,
            "use_preprocess": self.use_preprocess,
            "scale": self.scale,
            "random_state": self.random_state,
            "model": self.model,
            "preprocessor": self.preprocessor,
            "trained": self.trained,
        }

        save_object(package, path)

        if self.verbose:
            print(f"Model saved to {path}")

    # ================= LOAD =================
    @classmethod
    def load(cls, path, verbose=True):
        """Load AutoModel from disk."""
        package = load_object(path)

        obj = cls(
            task=package["task"],
            preprocess=package["use_preprocess"],
            scale=package["scale"],
            verbose=verbose,
            random_state=package["random_state"],
        )

        obj.model = package["model"]
        obj.preprocessor = package["preprocessor"]
        obj.columns = package["columns"]
        obj.trained = package["trained"]

        if verbose:
            model_name = type(obj.model).__name__
            feature_count = len(obj.columns)

            print(f"Loaded {model_name} "f"({obj.task}) | Features: {feature_count}")

        return obj