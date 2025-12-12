import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys

if __name__ == "__main__":
    # Ambil parameter input dari MLproject
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    dataset_path = sys.argv[3]

    # Load dataset
    data = pd.read_csv(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        test_size=0.2,
        random_state=42
    )

    input_example = X_train[:5]

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        # Logging
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        print("Training selesai!")
        print("Akurasi:", acc)
