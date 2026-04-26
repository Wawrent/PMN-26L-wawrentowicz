from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import build_model
from src.experiments import run_experiments
from src.visualization import plot_roc


def main():
    df = load_data("data/heart.csv")

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    model = build_model(preprocessor)
    model.fit(X_train, y_train)

    run_experiments(build_model, preprocessor, X_train, X_test, y_train, y_test)

    plot_roc(model, X_test, y_test)


if __name__ == "__main__":
    main()