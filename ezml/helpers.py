from .automodel import AutoModel


def train_model(
    data,
    target,
    task="auto",
    mode="fast",   # ⭐ NEW
    preprocess=True,
    scale=False,
    verbose=True,
    random_state=42,
):
    """
    One-line training helper for ezml.
    """

    model = AutoModel(
        task=task,
        mode=mode,
        preprocess=preprocess,
        scale=scale,
        verbose=verbose,
        random_state=random_state,
    )

    model.train(data, target=target)
    return model