import json
import pandas as pd

import mlflow
import mlflow.xgboost

import dagshub
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# DAGSHUB + MLFLOW CONFIG
dagshub.init(
    repo_owner="ayughaniyatur",
    repo_name="Membangun_Model_Ayu_Ghaniyatur",
    mlflow=True
)

mlflow.set_experiment("Customer_Churn_Tuning")

# LOAD DATA
df = pd.read_csv("customer_churn_processed.csv")

TARGET_COLUMN = "Churn"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# XGBOOST MODEL
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


# HYPERPARAMETER TUNING
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("Starting GridSearchCV...")

grid_search.fit(X_train, y_train)

print("GridSearchCV Finished")


# BEST MODEL
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


# METRICS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


# MLFLOW LOGGING
with mlflow.start_run(run_name="XGBoost_Tuning"):

    # PARAMETERS
    mlflow.log_param("model_type", "XGBoost")

    for key, value in grid_search.best_params_.items():
        mlflow.log_param(key, value)

    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("cv", 3)
    mlflow.log_param("random_state", 42)

    # METRICS
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # MODEL
    mlflow.xgboost.log_model(xgb_model=best_model, artifact_path="model")

    # ARTIFACT 1: training_confusion_matrix
    plt.figure(figsize=(6, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Training Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig("training_confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("training_confusion_matrix.png")

    # ARTIFACT 2: metric_info.json
    metric_info = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    with open(
        "metric_info.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metric_info,
            f,
            indent=4
        )

    mlflow.log_artifact("metric_info.json")

    # ARTIFACT 3: estimator.html
    estimator_html = f"""
    <html>
    <head>
        <title>XGBoost Estimator</title>
    </head>
    <body>

        <h1>XGBoost Best Estimator</h1>

        <pre>
        {best_model}
        </pre>

    </body>
    </html>
    """

    with open(
        "estimator.html",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(estimator_html)

    mlflow.log_artifact(
        "estimator.html"
    )

    # ARTIFACT 4: feature_importance.png
    plt.figure(figsize=(8, 6))

    xgb.plot_importance(best_model, max_num_features=10)

    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    mlflow.log_artifact("feature_importance.png")

    # ARTIFACT 5: best_params.json
    with open(
        "best_params.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            grid_search.best_params_,
            f,
            indent=4
        )

    mlflow.log_artifact("best_params.json")

    # OUTPUT
    print("\n===== TUNING RESULT =====")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nBest Parameters:")
    print(grid_search.best_params_)

print("\nXGBoost tuning selesai.")