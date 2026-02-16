import joblib


def save_object(obj, path):
    """Save object to disk."""
    joblib.dump(obj, path)


def load_object(path):
    """Load object from disk."""
    return joblib.load(path)