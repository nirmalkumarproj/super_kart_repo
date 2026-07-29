import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from huggingface_hub import HfApi, create_repo
import mlflow
import numpy as np
import os

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("SuperKart-Sales-Forecasting-Experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = "hf://datasets/nirmalhugface/super_kart/Xtrain.csv"
Xtest_path = "hf://datasets/nirmalhugface/super_kart/Xtest.csv"
ytrain_path = "hf://datasets/nirmalhugface/super_kart/ytrain.csv"
ytest_path = "hf://datasets/nirmalhugface/super_kart/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year"
]

categorical_features = [
    "Product_Id",
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

xgb_model = xgb.XGBRegressor(
    random_state=42
)

param_grid = {
    "xgbregressor__n_estimators": [50, 75, 100, 125, 150],
    "xgbregressor__max_depth": [2, 3, 4],
    "xgbregressor__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbregressor__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbregressor__learning_rate": [0.01, 0.05, 0.1],
    "xgbregressor__reg_lambda": [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error"
    )

    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_rmse = np.sqrt(mean_squared_error(ytrain, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(ytest, y_pred_test))

    train_mae = mean_absolute_error(ytrain, y_pred_train)
    test_mae = mean_absolute_error(ytest, y_pred_test)

    train_r2 = r2_score(ytrain, y_pred_train)
    test_r2 = r2_score(ytest, y_pred_test)

    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

    model_path = "best_super_kart_sales_model_v1.joblib"

    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved as artifact at: {model_path}")

    repo_id = "nirmalhugface/super_kart_sales_model"
    repo_type = "model"

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=os.getenv("HF_TOKEN"),
        exist_ok=True
    )

    print(f"Model repository '{repo_id}' is ready.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

    print("Model uploaded successfully.")
