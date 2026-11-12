"""
src/02_train_and_explain.py
===========================
Rigorous Machine Learning, Deep Learning, Bayesian Optimization, Uncertainty Quantification,
and Explainable AI (XAI) pipeline for Bengaluru EV Station Congestion Prediction.

Evaluates 5 Classical ML Algorithms + 2 Advanced AI/ML Techniques:
1. Logistic Regression / Ridge Regression
2. Random Forest (Classifier & Regressor)
3. Support Vector Machine (SVC & SVR with RBF kernel)
4. XGBoost (Classifier & Regressor)
5. k-Nearest Neighbors (Classifier & Regressor)
6. Deep Learning: PyTorch Bi-LSTM (Multi-Task sequence modeling)
7. Bayesian Optimization (Optuna tuned XGBoost) + LightGBM Quantile Regressors (80% Prediction Intervals)
8. Explainable AI: Global TreeSHAP Summary + Local SHAP Waterfall Plot
"""

import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, mean_squared_error,
    mean_absolute_error, r2_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize

import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "bengaluru_ev_fused.csv"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "results" / "plots"
XAI_DIR = BASE_DIR / "results" / "xai"
BENCHMARK_PATH = BASE_DIR / "results" / "model_benchmark.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
XAI_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. DATA LOADING & CHRONOLOGICAL SPLIT
# -----------------------------------------------------------------------------
def load_and_split_data():
    print("=" * 80)
    print("STEP 1: CHRONOLOGICAL 80/20 TRAIN-TEST SPLIT (NO DATA LEAKAGE)")
    print("=" * 80)
    print(f"Loading fused dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    n_total = len(df)
    split_idx = int(n_total * 0.80)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    train_start, train_end = train_df["timestamp"].min(), train_df["timestamp"].max()
    test_start, test_end = test_df["timestamp"].min(), test_df["timestamp"].max()
    
    print(f"Total Observations: {n_total:,}")
    print(f"  Training Split : {len(train_df):,} rows (80.0%) | {train_start} -> {train_end}")
    print(f"  Testing Split  : {len(test_df):,} rows (20.0%) | {test_start} -> {test_end}")
    
    return train_df, test_df


# -----------------------------------------------------------------------------
# 2. FEATURE SELECTION (MUTUAL INFORMATION)
# -----------------------------------------------------------------------------
def perform_feature_selection(train_df):
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE SELECTION VIA MUTUAL INFORMATION CLASSIF")
    print("=" * 80)
    
    candidate_features = [
        "latitude", "longitude", "total_ports", "utilization_rate",
        "temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_peak_commute",
        "util_lag_30m", "util_lag_60m", "util_lag_120m", "util_lag_24h",
        "util_roll_mean_3h", "util_roll_std_3h", "temp_roll_mean_3h",
        "precip_roll_mean_3h", "temp_x_util", "rain_x_peak_hour"
    ]
    
    # Subsample up to 15,000 training observations stratified by congestion_state for fast MI calculation
    sample_indices = []
    for _, grp in train_df.groupby("congestion_state"):
        sample_indices.extend(grp.sample(n=min(len(grp), 5000), random_state=42).index)
    sample_train = train_df.loc[sample_indices]
    
    X_mi = sample_train[candidate_features].values
    y_mi = sample_train["congestion_state"].values
    
    print(f"Computing Mutual Information across {len(candidate_features)} candidate features on {len(sample_train):,} samples...")
    mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42, n_neighbors=5)
    
    mi_df = pd.DataFrame({
        "feature": candidate_features,
        "mi_score": mi_scores
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)
    
    top_15_features = mi_df["feature"].head(15).tolist()
    
    # Save top 15 features to JSON
    feature_meta = {
        "selected_features": top_15_features,
        "all_mi_rankings": mi_df.to_dict(orient="records")
    }
    with open(MODELS_DIR / "selected_features.json", "w") as f:
        json.dump(feature_meta, f, indent=2)
        
    print(f"Top 15 Features Selected & Saved to {MODELS_DIR / 'selected_features.json'}:")
    for rank, row in mi_df.head(15).iterrows():
        print(f"  {rank+1:2d}. {row['feature']:<24}: MI Score = {row['mi_score']:.4f}")
        
    return top_15_features


# -----------------------------------------------------------------------------
# 3. METRICS EVALUATION HELPER
# -----------------------------------------------------------------------------
def evaluate_predictions(name, y_true_clf, y_pred_clf, y_proba_clf,
                         y_true_reg, y_pred_reg, latency_ms):
    """Calculates comprehensive classification and regression metrics."""
    acc = accuracy_score(y_true_clf, y_pred_clf)
    prec = precision_score(y_true_clf, y_pred_clf, average="macro", zero_division=0)
    rec = recall_score(y_true_clf, y_pred_clf, average="macro", zero_division=0)
    f1 = f1_score(y_true_clf, y_pred_clf, average="macro", zero_division=0)
    
    try:
        auc_score = roc_auc_score(y_true_clf, y_proba_clf, multi_class="ovr", average="macro")
    except Exception:
        auc_score = np.nan
        
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)
    
    metrics = {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Macro_F1": round(f1, 4),
        "ROC_AUC": round(auc_score, 4) if not np.isnan(auc_score) else None,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Inference_Latency_ms": round(latency_ms, 2)
    }
    return metrics


# -----------------------------------------------------------------------------
# 4. CLASSICAL ML ALGORITHMS
# -----------------------------------------------------------------------------
def train_classical_models(X_train, y_train_clf, y_train_reg,
                           X_test, y_test_clf, y_test_reg,
                           X_train_scaled, X_test_scaled, scaler):
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING 5 CLASSICAL ML ALGORITHMS")
    print("=" * 80)
    
    benchmark_results = []
    trained_artifacts = {}
    
    # -------------------------------------------------------------------------
    # Model 1: Logistic Regression (L2) / Ridge Regression
    # -------------------------------------------------------------------------
    print("\n[1/5] Training Logistic Regression (L2) & Ridge Regression...")
    clf_lr = LogisticRegression(penalty="l2", C=1.0, max_iter=500, random_state=42)
    reg_ridge = Ridge(alpha=1.0, random_state=42)
    
    clf_lr.fit(X_train_scaled, y_train_clf)
    reg_ridge.fit(X_train_scaled, y_train_reg)
    
    t0 = time.time()
    pred_clf_lr = clf_lr.predict(X_test_scaled)
    proba_clf_lr = clf_lr.predict_proba(X_test_scaled)
    pred_reg_ridge = np.clip(reg_ridge.predict(X_test_scaled), 0.0, 1.0)
    t_lat = ((time.time() - t0) * 1000.0 / len(X_test)) * 1000.0  # ms per 1k samples
    
    m1_metrics = evaluate_predictions(
        "Logistic / Ridge Regression", y_test_clf, pred_clf_lr, proba_clf_lr,
        y_test_reg, pred_reg_ridge, t_lat
    )
    benchmark_results.append(m1_metrics)
    joblib.dump(clf_lr, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(reg_ridge, MODELS_DIR / "ridge_regression.joblib")
    trained_artifacts["lr_clf"] = (pred_clf_lr, proba_clf_lr)
    print(f"  -> Accuracy: {m1_metrics['Accuracy']:.4f} | Macro F1: {m1_metrics['Macro_F1']:.4f} | RMSE: {m1_metrics['RMSE']:.4f}")

    # -------------------------------------------------------------------------
    # Model 2: Random Forest (Classifier & Regressor)
    # -------------------------------------------------------------------------
    print("\n[2/5] Training Random Forest (Classifier & Regressor, n_jobs=-1)...")
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42)
    reg_rf = RandomForestRegressor(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42)
    
    clf_rf.fit(X_train, y_train_clf)
    reg_rf.fit(X_train, y_train_reg)
    
    t0 = time.time()
    pred_clf_rf = clf_rf.predict(X_test)
    proba_clf_rf = clf_rf.predict_proba(X_test)
    pred_reg_rf = np.clip(reg_rf.predict(X_test), 0.0, 1.0)
    t_lat = ((time.time() - t0) * 1000.0 / len(X_test)) * 1000.0
    
    m2_metrics = evaluate_predictions(
        "Random Forest", y_test_clf, pred_clf_rf, proba_clf_rf,
        y_test_reg, pred_reg_rf, t_lat
    )
    benchmark_results.append(m2_metrics)
    joblib.dump(clf_rf, MODELS_DIR / "random_forest_clf.joblib")
    joblib.dump(reg_rf, MODELS_DIR / "random_forest_reg.joblib")
    trained_artifacts["rf_clf"] = (pred_clf_rf, proba_clf_rf)
    print(f"  -> Accuracy: {m2_metrics['Accuracy']:.4f} | Macro F1: {m2_metrics['Macro_F1']:.4f} | RMSE: {m2_metrics['RMSE']:.4f}")

    # -------------------------------------------------------------------------
    # Model 3: Support Vector Machine (SVC & SVR with RBF Kernel on 15,000 sub-sample)
    # -------------------------------------------------------------------------
    print("\n[3/5] Training Support Vector Machine (SVC & SVR RBF on 15,000-sample subsample)...")
    np.random.seed(42)
    sub_indices = np.random.choice(len(X_train_scaled), size=15000, replace=False)
    X_sub_scaled = X_train_scaled[sub_indices]
    y_sub_clf = y_train_clf[sub_indices]
    y_sub_reg = y_train_reg[sub_indices]
    
    clf_svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=42, cache_size=500)
    reg_svm = SVR(kernel="rbf", C=1.0, cache_size=500)
    
    clf_svm.fit(X_sub_scaled, y_sub_clf)
    reg_svm.fit(X_sub_scaled, y_sub_reg)
    
    t0 = time.time()
    pred_clf_svm = clf_svm.predict(X_test_scaled)
    proba_clf_svm = clf_svm.predict_proba(X_test_scaled)
    pred_reg_svm = np.clip(reg_svm.predict(X_test_scaled), 0.0, 1.0)
    t_lat = ((time.time() - t0) * 1000.0 / len(X_test)) * 1000.0
    
    m3_metrics = evaluate_predictions(
        "Support Vector Machine (RBF)", y_test_clf, pred_clf_svm, proba_clf_svm,
        y_test_reg, pred_reg_svm, t_lat
    )
    benchmark_results.append(m3_metrics)
    joblib.dump(clf_svm, MODELS_DIR / "svm_clf.joblib")
    joblib.dump(reg_svm, MODELS_DIR / "svm_reg.joblib")
    trained_artifacts["svm_clf"] = (pred_clf_svm, proba_clf_svm)
    print(f"  -> Accuracy: {m3_metrics['Accuracy']:.4f} | Macro F1: {m3_metrics['Macro_F1']:.4f} | RMSE: {m3_metrics['RMSE']:.4f}")

    # -------------------------------------------------------------------------
    # Model 4: XGBoost (Baseline, tree_method='hist')
    # -------------------------------------------------------------------------
    print("\n[4/5] Training XGBoost Baseline (Classifier & Regressor, tree_method='hist')...")
    clf_xgb = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        tree_method="hist", random_state=42, n_jobs=-1
    )
    reg_xgb = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        tree_method="hist", random_state=42, n_jobs=-1
    )
    
    clf_xgb.fit(X_train, y_train_clf)
    reg_xgb.fit(X_train, y_train_reg)
    
    t0 = time.time()
    pred_clf_xgb = clf_xgb.predict(X_test)
    proba_clf_xgb = clf_xgb.predict_proba(X_test)
    pred_reg_xgb = np.clip(reg_xgb.predict(X_test), 0.0, 1.0)
    t_lat = ((time.time() - t0) * 1000.0 / len(X_test)) * 1000.0
    
    m4_metrics = evaluate_predictions(
        "XGBoost (Baseline)", y_test_clf, pred_clf_xgb, proba_clf_xgb,
        y_test_reg, pred_reg_xgb, t_lat
    )
    benchmark_results.append(m4_metrics)
    joblib.dump(clf_xgb, MODELS_DIR / "xgboost_clf.joblib")
    joblib.dump(reg_xgb, MODELS_DIR / "xgboost_reg.joblib")
    trained_artifacts["xgb_base_clf"] = (pred_clf_xgb, proba_clf_xgb)
    print(f"  -> Accuracy: {m4_metrics['Accuracy']:.4f} | Macro F1: {m4_metrics['Macro_F1']:.4f} | RMSE: {m4_metrics['RMSE']:.4f}")

    # -------------------------------------------------------------------------
    # Model 5: k-Nearest Neighbors (k-NN, n_jobs=-1 on 20,000 sub-sample)
    # -------------------------------------------------------------------------
    print("\n[5/5] Training k-Nearest Neighbors (k=7, n_jobs=-1 on 20,000 subsample)...")
    knn_indices = np.random.choice(len(X_train_scaled), size=20000, replace=False)
    X_knn_train = X_train_scaled[knn_indices]
    y_knn_clf = y_train_clf[knn_indices]
    y_knn_reg = y_train_reg[knn_indices]
    
    clf_knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    reg_knn = KNeighborsRegressor(n_neighbors=7, n_jobs=-1)
    
    clf_knn.fit(X_knn_train, y_knn_clf)
    reg_knn.fit(X_knn_train, y_knn_reg)
    
    t0 = time.time()
    pred_clf_knn = clf_knn.predict(X_test_scaled)
    proba_clf_knn = clf_knn.predict_proba(X_test_scaled)
    pred_reg_knn = np.clip(reg_knn.predict(X_test_scaled), 0.0, 1.0)
    t_lat = ((time.time() - t0) * 1000.0 / len(X_test)) * 1000.0
    
    m5_metrics = evaluate_predictions(
        "k-Nearest Neighbors (k=7)", y_test_clf, pred_clf_knn, proba_clf_knn,
        y_test_reg, pred_reg_knn, t_lat
    )
    benchmark_results.append(m5_metrics)
    joblib.dump(clf_knn, MODELS_DIR / "knn_clf.joblib")
    joblib.dump(reg_knn, MODELS_DIR / "knn_reg.joblib")
    trained_artifacts["knn_clf"] = (pred_clf_knn, proba_clf_knn)
    print(f"  -> Accuracy: {m5_metrics['Accuracy']:.4f} | Macro F1: {m5_metrics['Macro_F1']:.4f} | RMSE: {m5_metrics['RMSE']:.4f}")
    
    return benchmark_results, trained_artifacts


# -----------------------------------------------------------------------------
# 5. ADVANCED TECHNIQUE 1: PYTORCH BI-LSTM (MULTI-TASK SEQUENCES)
# -----------------------------------------------------------------------------
class MultiTaskBiLSTM(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=48, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        self.clf_head = nn.Linear(64, num_classes)
        self.reg_head = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        final_rep = self.fc_shared(out[:, -1, :])
        logits = self.clf_head(final_rep)
        reg = torch.sigmoid(self.reg_head(final_rep)).squeeze(-1)
        return logits, reg


def train_pytorch_bilstm(train_df, test_df, selected_features, scaler):
    print("\n" + "=" * 80)
    print("STEP 4: ADVANCED TECHNIQUE 1 - PYTORCH BI-LSTM (6-STEP / 3-HOUR SEQUENCES)")
    print("=" * 80)
    
    seq_len = 6  # 6 steps of 30 mins = 3 hours
    
    def extract_sequences(df):
        X_seq, y_clf, y_reg = [], [], []
        for _, group in df.groupby("station_id", sort=False):
            scaled_vals = scaler.transform(group[selected_features].values)
            clfs = group["congestion_state"].values
            regs = group["future_utilization_30m"].values
            n = len(scaled_vals)
            for i in range(seq_len - 1, n):
                X_seq.append(scaled_vals[i - seq_len + 1 : i + 1])
                y_clf.append(clfs[i])
                y_reg.append(regs[i])
        return (
            torch.tensor(np.array(X_seq), dtype=torch.float32),
            torch.tensor(np.array(y_clf), dtype=torch.long),
            torch.tensor(np.array(y_reg), dtype=torch.float32)
        )
        
    print("Formulating sliding temporal sequences per station...")
    X_train_seq, y_train_clf_seq, y_train_reg_seq = extract_sequences(train_df)
    X_test_seq, y_test_clf_seq, y_test_reg_seq = extract_sequences(test_df)
    print(f"Sequence Dimensions: Train = {X_train_seq.shape}, Test = {X_test_seq.shape}")
    
    train_dataset = TensorDataset(X_train_seq, y_train_clf_seq, y_train_reg_seq)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    model = MultiTaskBiLSTM(input_dim=len(selected_features), hidden_dim=48, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    criterion_clf = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    epochs = 6
    print(f"Training MultiTask Bi-LSTM for {epochs} epochs on CPU...")
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for bx, by_clf, by_reg in train_loader:
            optimizer.zero_grad()
            logits, pred_reg = model(bx)
            loss = criterion_clf(logits, by_clf) + 4.0 * criterion_reg(pred_reg, by_reg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        avg_loss = total_loss / len(train_dataset)
        print(f"  Epoch {ep+1}/{epochs} | Multi-task Loss: {avg_loss:.4f}")
        
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        test_logits, test_regs = model(X_test_seq)
        test_probas = torch.softmax(test_logits, dim=-1).numpy()
        test_preds_clf = np.argmax(test_probas, axis=-1)
        test_preds_reg = np.clip(test_regs.numpy(), 0.0, 1.0)
    t_lat = ((time.time() - t0) * 1000.0 / len(X_test_seq)) * 1000.0
    
    metrics = evaluate_predictions(
        "PyTorch Bi-LSTM",
        y_test_clf_seq.numpy(), test_preds_clf, test_probas,
        y_test_reg_seq.numpy(), test_preds_reg, t_lat
    )
    print(f"  -> Accuracy: {metrics['Accuracy']:.4f} | Macro F1: {metrics['Macro_F1']:.4f} | RMSE: {metrics['RMSE']:.4f}")
    
    torch.save(model.state_dict(), MODELS_DIR / "bilstm_multitask.pth")
    return metrics, (test_preds_clf, test_probas, y_test_clf_seq.numpy())


# -----------------------------------------------------------------------------
# 6. ADVANCED TECHNIQUE 2: BAYESIAN OPTIMIZATION & QUANTILE REGRESSION
# -----------------------------------------------------------------------------
def run_bayesian_optimization_and_quantiles(X_train, y_train_clf, y_train_reg,
                                            X_test, y_test_clf, y_test_reg,
                                            selected_features):
    print("\n" + "=" * 80)
    print("STEP 5: ADVANCED TECHNIQUE 2 - BAYESIAN OPTIMIZATION & UNCERTAINTY QUANTIFICATION")
    print("=" * 80)
    
    # 1. Optuna Hyperparameter Optimization (20 Trials) for XGBoost
    print("Running Optuna (20 Trials) for XGBoost Tuning...")
    n_tr = len(X_train)
    val_split = int(n_tr * 0.80)
    X_tr, y_tr_c = X_train[:val_split], y_train_clf[:val_split]
    X_val, y_val_c = X_train[val_split:], y_train_clf[val_split:]
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 180),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.22, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr_c)
        preds = model.predict(X_val)
        return f1_score(y_val_c, preds, average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    print(f"Optuna Best Macro F1 (Val): {study.best_value:.4f}")
    print(f"Optimal Parameters: {best_params}")
    
    # Train full tuned models
    tuned_params = best_params.copy()
    tuned_params.update({"tree_method": "hist", "random_state": 42, "n_jobs": -1})
    
    tuned_clf = xgb.XGBClassifier(**tuned_params)
    tuned_reg = xgb.XGBRegressor(**tuned_params)
    
    tuned_clf.fit(X_train, y_train_clf)
    tuned_reg.fit(X_train, y_train_reg)
    
    t0 = time.time()
    pred_tuned_clf = tuned_clf.predict(X_test)
    proba_tuned_clf = tuned_clf.predict_proba(X_test)
    pred_tuned_reg = np.clip(tuned_reg.predict(X_test), 0.0, 1.0)
    t_lat = ((time.time() - t0) * 1000.0 / len(X_test)) * 1000.0
    
    tuned_metrics = evaluate_predictions(
        "Tuned XGBoost (Optuna)", y_test_clf, pred_tuned_clf, proba_tuned_clf,
        y_test_reg, pred_tuned_reg, t_lat
    )
    joblib.dump(tuned_clf, MODELS_DIR / "xgboost_tuned_clf.joblib")
    joblib.dump(tuned_reg, MODELS_DIR / "xgboost_tuned_reg.joblib")
    print(f"  -> Tuned XGBoost Accuracy: {tuned_metrics['Accuracy']:.4f} | Macro F1: {tuned_metrics['Macro_F1']:.4f} | RMSE: {tuned_metrics['RMSE']:.4f}")
    
    # 2. Uncertainty Quantification: LightGBM Quantile Regressors (alpha = 0.10, 0.50, 0.90)
    print("\nTraining LightGBM Quantile Regressors (alpha = 0.10, 0.50, 0.90) for 80% Prediction Intervals...")
    quantiles = [0.10, 0.50, 0.90]
    q_models = {}
    q_preds = {}
    
    for alpha in quantiles:
        print(f"  Fitting Quantile Regressor (alpha={alpha:.2f})...")
        q_lgb = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=100,
            learning_rate=0.08,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        q_lgb.fit(X_train, y_train_reg)
        q_models[alpha] = q_lgb
        q_preds[alpha] = np.clip(q_lgb.predict(X_test), 0.0, 1.0)
        joblib.dump(q_lgb, MODELS_DIR / f"lgbm_quantile_{int(alpha*100)}.joblib")
        
    q_lower = q_preds[0.10]
    q_median = q_preds[0.50]
    q_upper = q_preds[0.90]
    
    in_interval = (y_test_reg >= q_lower) & (y_test_reg <= q_upper)
    coverage = np.mean(in_interval) * 100.0
    mean_width = np.mean(q_upper - q_lower)
    print(f"\nUncertainty Calibration Results:")
    print(f"  Target Coverage  : 80.00%")
    print(f"  Empirical Coverage: {coverage:.2f}%")
    print(f"  Mean Interval Width: {mean_width:.4f}")
    
    t0 = time.time()
    _ = q_models[0.50].predict(X_test)
    t_lat_q = ((time.time() - t0) * 1000.0 / len(X_test)) * 1000.0
    
    q_clf_preds = np.select(
        [q_median < 0.50, (q_median >= 0.50) & (q_median <= 0.80)],
        [0, 1], default=2
    )
    q_probas = np.zeros((len(X_test), 3))
    for i, p in enumerate(q_median):
        if p < 0.50:
            q_probas[i, 0] = 0.80; q_probas[i, 1] = 0.18; q_probas[i, 2] = 0.02
        elif p <= 0.80:
            q_probas[i, 0] = 0.15; q_probas[i, 1] = 0.75; q_probas[i, 2] = 0.10
        else:
            q_probas[i, 0] = 0.02; q_probas[i, 1] = 0.18; q_probas[i, 2] = 0.80
            
    q_metrics = evaluate_predictions(
        "Quantile Regressor (LightGBM)", y_test_clf, q_clf_preds, q_probas,
        y_test_reg, q_median, t_lat_q
    )
    
    return tuned_metrics, q_metrics, tuned_clf, tuned_reg, (q_lower, q_median, q_upper, coverage, mean_width), (pred_tuned_clf, proba_tuned_clf)


# -----------------------------------------------------------------------------
# 7. GENERATE BENCHMARK ARTIFACTS & EVALUATION PLOTS
# -----------------------------------------------------------------------------
def generate_evaluation_plots(y_test_clf, y_test_reg,
                              rf_preds, xgb_tuned_preds, bilstm_preds,
                              quantile_info, test_df):
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING EVALUATION PLOTS IN results/plots/")
    print("=" * 80)
    
    sns.set_theme(style="whitegrid", font="sans-serif")
    class_labels = ["Low (< 0.50)", "Moderate (0.50-0.80)", "Congested (> 0.80)"]
    
    # -------------------------------------------------------------------------
    # Plot 1: Confusion Matrices Subplot (Tuned XGBoost, Random Forest, Bi-LSTM)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models_to_plot = [
        ("Tuned XGBoost", xgb_tuned_preds[0], y_test_clf),
        ("Random Forest", rf_preds[0], y_test_clf),
        ("PyTorch Bi-LSTM", bilstm_preds[0], bilstm_preds[2])
    ]
    
    for ax, (title, p_pred, p_true) in zip(axes, models_to_plot):
        cm = confusion_matrix(p_true, p_pred, normalize="true")
        sns.heatmap(
            cm, annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax,
            xticklabels=class_labels, yticklabels=class_labels
        )
        ax.set_title(f"{title}\nNormalized Confusion Matrix", weight="bold", pad=10)
        ax.set_xlabel("Predicted Congestion State")
        ax.set_ylabel("True Congestion State")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        
    plt.tight_layout()
    p1_path = PLOTS_DIR / "01_confusion_matrices.png"
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"  [1/3] Saved Confusion Matrices: {p1_path.name}")
    
    # -------------------------------------------------------------------------
    # Plot 2: Multi-Class ROC Curves (One-vs-Rest for Tuned XGBoost)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 7))
    y_test_bin = label_binarize(y_test_clf, classes=[0, 1, 2])
    proba_xgb = xgb_tuned_preds[1]
    
    colors = ["#2b5c8f", "#e08214", "#d73027"]
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba_xgb[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2.5,
                 label=f"ROC: {class_labels[i]} (AUC = {roc_auc:.3f})")
                 
    macro_auc = roc_auc_score(y_test_clf, proba_xgb, multi_class="ovr", average="macro")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level (AUC = 0.500)")
    plt.title(f"Multi-Class ROC Curves - Tuned XGBoost (Macro-Average AUC = {macro_auc:.3f})", pad=15, weight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    p2_path = PLOTS_DIR / "02_multiclass_roc_curves.png"
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"  [2/3] Saved ROC Curves: {p2_path.name}")
    
    # -------------------------------------------------------------------------
    # Plot 3: Quantile Prediction Intervals (Time Series Slice)
    # -------------------------------------------------------------------------
    q_lower, q_median, q_upper, coverage, mean_width = quantile_info
    plt.figure(figsize=(15, 6))
    
    slice_len = 120
    test_slice_idx = np.arange(slice_len)
    
    plt.plot(test_slice_idx, y_test_reg[:slice_len], "o-", color="#1a1a1a", lw=1.8, markersize=3.5, label="Ground Truth Utilization")
    plt.plot(test_slice_idx, q_median[:slice_len], "--", color="#02818a", lw=2.2, label="Median Prediction (q=0.50)")
    plt.fill_between(
        test_slice_idx, q_lower[:slice_len], q_upper[:slice_len],
        color="#67a9cf", alpha=0.38, label=f"Calibrated 80% Prediction Interval [q10, q90]"
    )
    
    plt.axhline(0.80, color="#d73027", linestyle=":", lw=1.5, label="Congested Threshold (0.80)")
    plt.axhline(0.50, color="#fec44f", linestyle=":", lw=1.5, label="Moderate Threshold (0.50)")
    
    plt.title(f"Bengaluru EV Utilization Uncertainty Quantification (Test Set Slice)\nEmpirical 80% Coverage = {coverage:.2f}% | Mean Interval Width = {mean_width:.3f}", pad=15, weight="bold")
    plt.xlabel("Chronological Time Steps (30-Minute Intervals)")
    plt.ylabel("Station Utilization Rate")
    plt.ylim(-0.05, 1.15)
    plt.legend(loc="upper right", ncol=2, frameon=True)
    plt.tight_layout()
    p3_path = PLOTS_DIR / "03_quantile_prediction_intervals.png"
    plt.savefig(p3_path, dpi=300)
    plt.close()
    print(f"  [3/3] Saved Quantile Prediction Intervals: {p3_path.name}")


# -----------------------------------------------------------------------------
# 8. EXPLAINABLE AI (XAI) VIA TREESHAP
# -----------------------------------------------------------------------------
def run_explainable_ai(tuned_reg, X_test, selected_features, y_test_clf):
    print("\n" + "=" * 80)
    print("STEP 7: EXPLAINABLE AI (XAI) VIA TREESHAP ON 2,000 TEST SAMPLES")
    print("=" * 80)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=2000, replace=False)
    X_shap_sample = X_test[sample_indices]
    
    feature_df = pd.DataFrame(X_shap_sample, columns=selected_features)
    
    print(f"Initializing TreeExplainer on Tuned XGBoost Regressor...")
    explainer = shap.TreeExplainer(tuned_reg)
    shap_values = explainer(feature_df)
    
    # Plot 1: Global SHAP Summary Plot
    p1_path = XAI_DIR / "01_shap_summary_plot.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_df, show=False)
    plt.title("Global TreeSHAP Feature Attribution (Impact on 30m Utilization)", pad=15, weight="bold")
    plt.tight_layout()
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"  [1/2] Saved Global SHAP Summary Plot: {p1_path.name}")
    
    # Plot 2: Local SHAP Waterfall Plot for an actual Congested Test Instance
    congested_candidates = np.where(y_test_clf[sample_indices] == 2)[0]
    if len(congested_candidates) > 0:
        target_idx = congested_candidates[0]
    else:
        target_idx = 0
        
    p2_path = XAI_DIR / "02_shap_waterfall_congested.png"
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[target_idx], show=False)
    plt.title("Local SHAP Attribution for Congested EV Station State (>0.80)", pad=15, weight="bold")
    plt.tight_layout()
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"  [2/2] Saved Local SHAP Waterfall Plot: {p2_path.name}")


# -----------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION
# -----------------------------------------------------------------------------
def main():
    start_wall = time.time()
    
    # 1. Load and Chronological Split
    train_df, test_df = load_and_split_data()
    
    # 2. Feature Selection
    selected_features = perform_feature_selection(train_df)
    
    # 3. Prepare Feature Matrices
    X_train = train_df[selected_features].values
    y_train_clf = train_df["congestion_state"].values
    y_train_reg = train_df["future_utilization_30m"].values
    
    X_test = test_df[selected_features].values
    y_test_clf = test_df["congestion_state"].values
    y_test_reg = test_df["future_utilization_30m"].values
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    
    # 4. Classical ML Algorithms
    benchmark_list, trained_artifacts = train_classical_models(
        X_train, y_train_clf, y_train_reg,
        X_test, y_test_clf, y_test_reg,
        X_train_scaled, X_test_scaled, scaler
    )
    
    # 5. Advanced Technique 1: PyTorch Bi-LSTM
    bilstm_metrics, bilstm_preds = train_pytorch_bilstm(train_df, test_df, selected_features, scaler)
    benchmark_list.append(bilstm_metrics)
    
    # 6. Advanced Technique 2: Optuna Tuning & Quantile Regression
    tuned_metrics, q_metrics, tuned_clf, tuned_reg, quantile_info, xgb_tuned_preds = (
        run_bayesian_optimization_and_quantiles(
            X_train, y_train_clf, y_train_reg,
            X_test, y_test_clf, y_test_reg,
            selected_features
        )
    )
    benchmark_list.append(tuned_metrics)
    benchmark_list.append(q_metrics)
    
    # 7. Generate Evaluation Plots
    generate_evaluation_plots(
        y_test_clf, y_test_reg,
        trained_artifacts["rf_clf"], xgb_tuned_preds, bilstm_preds,
        quantile_info, test_df
    )
    
    # 8. Explainable AI (XAI)
    run_explainable_ai(tuned_reg, X_test, selected_features, y_test_clf)
    
    # 9. Final Benchmark Summary Table
    benchmark_df = pd.DataFrame(benchmark_list)
    benchmark_df = benchmark_df.sort_values(by="Macro_F1", ascending=False).reset_index(drop=True)
    benchmark_df.to_csv(BENCHMARK_PATH, index=False)
    
    elapsed = time.time() - start_wall
    print("\n" + "=" * 105)
    print(f"BENCHMARK RESULTS ACROSS ALL 8 MODELS (Saved to {BENCHMARK_PATH})")
    print(f"Total Pipeline Runtime: {elapsed:.1f} seconds")
    print("=" * 105)
    print(benchmark_df.to_string(index=False))
    print("=" * 105)


if __name__ == "__main__":
    main()
