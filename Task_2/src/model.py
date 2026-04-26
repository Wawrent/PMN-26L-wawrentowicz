from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model(preprocessor, C=1.0, solver="lbfgs", max_iter=1000):
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter
        ))
    ])
    return model