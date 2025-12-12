import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    # Ambil parameter input dari MLproject
    # Usage dari MLproject:
    # python modelling.py {n_estimators} {max_depth} train_pca.csv
    if len(sys.argv) < 4:
        raise ValueError(
            "Argumen kurang. Format: python modelling.py <n_estimators> <max_depth> <dataset_path>"
        )

    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    dataset_path = sys.argv[3]

    # Load dataset
    data = pd.read_csv(dataset_path)

    # Pastikan kolom target ada
    target_col = "Credit_Score"
    if target_col not in data.columns:
        raise ValueError(
            f"Kolom target '{target_col}' tidak ditemukan. Kolom yang ada: {list(data.columns)}"
        )

    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    input_example = X_train.head(5)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        # Logging params & metric
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        # Log model + input example
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        print("Training selesai!")
        print("Akurasi:", acc)


if __name__ == "__main__":
    main()
