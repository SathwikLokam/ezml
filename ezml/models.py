def get_model(task, mode="fast", random_state=42):
    """
    Return model based on task and mode.

    mode:
        - fast → RandomForest
        - best → LightGBM (fallback to RF if not available)
    """

    # ================= FAST MODE =================
    if mode == "fast":
        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
        )

        if task == "classification":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
            )

        return RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
        )

    # ================= BEST MODE =================
    if mode == "best":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor

            if task == "classification":
                return LGBMClassifier(random_state=random_state)

            return LGBMRegressor(random_state=random_state)

        except Exception:
            # graceful fallback
            from sklearn.ensemble import (
                RandomForestClassifier,
                RandomForestRegressor,
            )

            print("LightGBM not installed. Falling back to RandomForest.")

            if task == "classification":
                return RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                )

            return RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
            )

    raise ValueError(f"Unknown mode: {mode}")