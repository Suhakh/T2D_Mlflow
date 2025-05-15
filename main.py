
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("final_data_2.csv")
X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("T2D Prediction")

best_model = None
best_score = 0.0
best_model_name = ""

def train_and_log_model(model, model_name, params):
    global best_model, best_score, best_model_name

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        mlflow.sklearn.log_model(model, f"{model_name}_model")

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_model_name = model_name
            print(f"New best model: {best_model_name} | F1 Score: {best_score:.4f}")

        print(f"{model_name} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

# --- Baseline Models ---
train_and_log_model(
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "RandomForest",
    {"n_estimators": 100, "max_depth": 5}
)

train_and_log_model(
    LogisticRegression(solver="liblinear", max_iter=200),
    "LogisticRegression",
    {"solver": "liblinear", "max_iter": 200}
)

train_and_log_model(
    XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss'),
    "XGBoost",
    {"n_estimators": 100, "max_depth": 4}
)

# --- XGBoost Hyperopt Tuning ---
def objective_xgb(params):
    with mlflow.start_run(nested=True, run_name="XGBoost_Trial"):
        clf = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=float(params['learning_rate']),
            use_label_encoder=False,
            eval_metric='logloss'
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        f1 = f1_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metrics({"f1_score": f1})

        return {'loss': -f1, 'status': STATUS_OK}

xgb_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3)
}

print("Starting XGBoost hyperparameter tuning...")
with mlflow.start_run(run_name="Tuned_XGBoost"):
    trials = Trials()
    best_xgb_params = fmin(
        fn=objective_xgb,
        space=xgb_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    best_xgb_params = {
        'n_estimators': int(best_xgb_params['n_estimators']),
        'max_depth': int(best_xgb_params['max_depth']),
        'learning_rate': float(best_xgb_params['learning_rate'])
    }

    tuned_xgb = XGBClassifier(**best_xgb_params, use_label_encoder=False, eval_metric='logloss')
    tuned_xgb.fit(X_train, y_train)
    preds = tuned_xgb.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_params(best_xgb_params)
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    signature = infer_signature(X_test, preds)
    mlflow.sklearn.log_model(tuned_xgb, "best_xgb_model", signature=signature)

    if f1 > best_score:
        best_model = tuned_xgb
        best_model_name = "Tuned_XGBoost"
        best_score = f1

    print(f"Tuned XGBoost completed | F1 Score: {f1:.4f}")

# --- RandomForest Hyperopt Tuning ---
def tune_rf(params):
    with mlflow.start_run(nested=True, run_name="RF_Trial"):
        rf = RandomForestClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            random_state=42
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        f1 = f1_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metrics({"f1_score": f1})

        return {'loss': -f1, 'status': STATUS_OK}

rf_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
    'max_depth': hp.quniform('max_depth', 3, 10, 1)
}

print("Starting RandomForest hyperparameter tuning...")
with mlflow.start_run(run_name="Tuned_RandomForest"):
    trials = Trials()
    best_rf_params = fmin(
        fn=tune_rf,
        space=rf_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    best_rf_params = {
        'n_estimators': int(best_rf_params['n_estimators']),
        'max_depth': int(best_rf_params['max_depth'])
    }

    tuned_rf = RandomForestClassifier(**best_rf_params, random_state=42)
    tuned_rf.fit(X_train, y_train)
    preds = tuned_rf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_params(best_rf_params)
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    signature = infer_signature(X_test, preds)
    mlflow.sklearn.log_model(tuned_rf, "best_rf_model", signature=signature)

    if f1 > best_score:
        best_model = tuned_rf
        best_model_name = "Tuned_RandomForest"
        best_score = f1

    print(f"Tuned RandomForest completed | F1 Score: {f1:.4f}")

# --- LogisticRegression Hyperopt Tuning ---
def tune_lr(params):
    with mlflow.start_run(nested=True, run_name="LR_Trial"):
        lr = LogisticRegression(
            solver=params['solver'],
            max_iter=int(params['max_iter'])
        )
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        f1 = f1_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metrics({"f1_score": f1})

        return {'loss': -f1, 'status': STATUS_OK}

lr_space = {
    'solver': hp.choice('solver', ['liblinear', 'lbfgs']),
    'max_iter': hp.quniform('max_iter', 100, 500, 50)
}

print("Starting LogisticRegression hyperparameter tuning...")
with mlflow.start_run(run_name="Tuned_LogisticRegression"):
    trials = Trials()
    best_lr_params = fmin(
        fn=tune_lr,
        space=lr_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    best_lr_params = {
        'solver': ['liblinear', 'lbfgs'][best_lr_params['solver']],
        'max_iter': int(best_lr_params['max_iter'])
    }

    tuned_lr = LogisticRegression(**best_lr_params)
    tuned_lr.fit(X_train, y_train)
    preds = tuned_lr.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_params(best_lr_params)
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    signature = infer_signature(X_test, preds)
    mlflow.sklearn.log_model(tuned_lr, "best_lr_model", signature=signature)

    if f1 > best_score:
        best_model = tuned_lr
        best_model_name = "Tuned_LogisticRegression"
        best_score = f1

    print(f"Tuned LogisticRegression completed | F1 Score: {f1:.4f}")

# --- Final Best Model Registration ---
if best_model is not None:
    with mlflow.start_run(run_name=f"Final_Best_{best_model_name}"):
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        signature = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(best_model, f"final_best_{best_model_name}_model", signature=signature)

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/final_best_{best_model_name}_model"
        registered_model = mlflow.register_model(model_uri, "T2D_FinalBestModel")

        client = MlflowClient()
        client.transition_model_version_stage(
            name="T2D_FinalBestModel",
            version=registered_model.version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f"✅ Final Best Model Registered: {best_model_name} | F1 Score: {f1:.4f}")
