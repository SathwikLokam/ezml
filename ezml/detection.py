import pandas as pd
import numpy as np


def detect_task(y):
    """
    Robust hybrid task detection for tabular data.

    Strategy:
    - object dtype → classification
    - binary labels → classification
    - small discrete integer sets → classification
    - continuous numeric → regression
    """

    # ---------- object/string ----------
    if y.dtype == "object":
        return "classification"

    # ---------- numeric ----------
    if pd.api.types.is_numeric_dtype(y):

        unique_vals = y.nunique(dropna=True)
        total_vals = len(y)

        # ✅ binary classification (strong signal)
        if unique_vals == 2:
            return "classification"

        # ✅ small discrete integer labels
        if pd.api.types.is_integer_dtype(y):

            ratio = unique_vals / max(total_vals, 1)

            # few unique compared to size
            if unique_vals <= 20 and ratio < 0.2:

                # check if values are small integers
                max_val = y.max()
                min_val = y.min()

                if min_val >= 0 and max_val <= 100:
                    return "classification"

        # otherwise treat as regression
        return "regression"

    # fallback
    return "classification"