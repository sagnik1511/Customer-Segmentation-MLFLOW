# Libraries :

import json
import mlflow
import warnings
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from src.metrics import metric_values
from src.processing import encode, fill_missing
from sklearn.ensemble import RandomForestClassifier as rfc


if __name__ == "__main__":
    np.random.seed(42)
    warnings.filterwarnings("ignore")

    # Fetching file paths

    with open("../config/filepath.json") as f:
        paths = json.load(f)
        f.close()
    # Loading DataFrames

    train_df = pd.read_csv(paths["train_base_path"])
    test_df = pd.read_csv(paths["test_base_path"])

    # Processing X and Y

    train_df, test_df = fill_missing(train_df, test_df)
    train_df, test_df = encode(train_df, test_df)

    X_train = train_df[train_df.columns[:-2]]
    y_train = train_df.iloc[:, -1]
    X_test = test_df[test_df.columns[:-2]]
    y_test = test_df.iloc[:, -1]

    # Defining Model Parameters

    params = {
        "n_estimators": 80,
        "random_state": 42,
        "max_features": "sqrt"

    }

    # Starting the MLFLOW Run

    with mlflow.start_run():
        model = rfc(n_estimators=params['n_estimators'],
                    max_features=params['max_features'],
                    random_state=params['random_state']
                    )
        model.fit(X_train, y_train)

        # Fetching Scores

        train_scores = metric_values(X_train, y_train, model)
        test_scores = metric_values(X_test, y_test, model)

        print(f"RandomForestClassifier({params})")

        for metric, score in train_scores.items():
            print(f"Training {metric} : {'%.4f'%score}")
            mlflow.log_metric(f"training_{metric}", score)
        for metric, score in test_scores.items():
            print(f"Testing {metric} : {'%.4f'%score}")
            mlflow.log_metric(f"testing_{metric}", score)

        for key, value in params.items():
            mlflow.log_param(key, value)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifierModel")
        else:
            mlflow.sklearn.log_model(model, "model")
