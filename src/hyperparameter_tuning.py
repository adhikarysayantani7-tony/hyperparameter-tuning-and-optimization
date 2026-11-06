"""
Hyperparameter Tuning and Optimization Pipeline
Internship Track: Week 3 - Task 1
Author: Sayantani Adhikary
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_and_preprocess(filepath: str):
    """Load dataset, split features and target, and apply standard scaling."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=['DefaultRisk'])
    y = df['DefaultRisk']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def evaluate_model(model, X_test, y_test, model_name: str):
    """Calculate and display standard evaluation metrics."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    
    return {
        'Model Configuration': model_name,
        'Accuracy (%)': f"{acc * 100:.2f}%",
        'Precision (%)': f"{prec * 100:.2f}%",
        'Recall (%)': f"{rec * 100:.2f}%",
        'F1-Score': f"{f1:.4f}"
    }


def main():
    data_path = "data/dataset.csv"
    X, y = load_and_preprocess(data_path)

    # 80/20 train-test split with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    # 1. Baseline Model (Default Hyperparameters)
    base_rf = RandomForestClassifier(random_state=42)
    base_rf.fit(X_train, y_train)
    results.append(evaluate_model(base_rf, X_test, y_test, "Baseline (Untuned RF)"))

    # Hyperparameter search space
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # 2. RandomizedSearchCV
    start_rand = time.time()
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=8,
        cv=cv,
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    rand_time = time.time() - start_rand
    best_rand = random_search.best_estimator_
    rand_res = evaluate_model(best_rand, X_test, y_test, f"RandomizedSearchCV ({rand_time:.2f}s)")
    results.append(rand_res)

    # 3. GridSearchCV
    start_grid = time.time()
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start_grid
    best_grid = grid_search.best_estimator_
    grid_res = evaluate_model(best_grid, X_test, y_test, f"GridSearchCV ({grid_time:.2f}s)")
    results.append(grid_res)

    # Output Scorecard
    print("=" * 75)
    print("           HYPERPARAMETER TUNING & OPTIMIZATION SCORECARD          ")
    print("=" * 75)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    print("=" * 75)
    print(f"\n[OPTIMAL PARAMETERS (GridSearchCV)]: {grid_search.best_params_}")


if __name__ == "__main__":
    main()
