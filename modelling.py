import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLFLOW CONFIG 
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Customer_Churn_Baseline")

print("Tracking URI:", mlflow.get_tracking_uri())

# AUTOLOG (WAJIB BASIC)
mlflow.sklearn.autolog()

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

# TRAIN MODEL
with mlflow.start_run(run_name="RandomForest_Baseline"):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Hanya ditampilkan di terminal
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n===== BASELINE RESULT =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

print("\nBaseline model selesai.")