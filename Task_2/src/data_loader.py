import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]

    df = pd.read_csv(path, header=None, names=columns)

    return df


def basic_info(df: pd.DataFrame):
    print("Basic info:")
    print(df.info())
    print("\nDescription:")
    print(df.describe())


def check_missing_values(df: pd.DataFrame):
    print("\nMissing values:")
    print(df.isnull().sum())


def value_counts(df: pd.DataFrame):
    print("\nValue Count:")
    for col in df.columns:
        print(f"\nColumn: {col}")
        print(df[col].value_counts())