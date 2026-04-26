from sklearn.metrics import accuracy_score


def run_experiments(model_builder, preprocessor, X_train, X_test, y_train, y_test):
    results = []

    C_values = [0.01, 0.1, 1, 10, 100]

    for C in C_values:
        model = model_builder(preprocessor, C=C)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append((C, acc))
        print(f"C={C}, Accuracy={acc:.4f}")

    return results