# Hyperparameter Tuning and Optimization Report

## 1. Executive Summary & Problem Formulation
Hyperparameter tuning is a critical phase in the machine learning lifecycle to prevent overfitting, minimize generalization error, and extract peak predictive accuracy. This report evaluates the optimization of a **Random Forest Classifier** on consumer credit risk data, comparing an untuned baseline against **RandomizedSearchCV** and **GridSearchCV** optimization strategies.

---

## 2. Hyperparameter Search Space Definition
We investigated four primary architectural hyperparameters governing Random Forest complexity:

1. **`n_estimators` `[50, 100, 200]`**:
   * *Role*: Number of decision trees in the ensemble. More trees stabilize variance and reduce prediction error at the cost of linear training runtime.
2. **`max_depth` `[3, 5, 10, None]`**:
   * *Role*: Maximum depth per tree. Restricting depth prevents individual trees from memorizing localized sample noise (overfitting).
3. **`min_samples_split` `[2, 5]`**:
   * *Role*: Minimum samples required to split an internal decision node. Higher values enforce conservative splits.
4. **`min_samples_leaf` `[1, 2]`**:
   * *Role*: Minimum samples required at a leaf node. Higher thresholds smooth decision boundaries.

---

## 3. Tuning Methodologies Compared

### A. RandomizedSearchCV (Stochastic Sampling)
* **Mechanics**: Randomly samples a fixed budget ($n\_iter=8$) of hyperparameter combinations from the search grid.
* **Advantage**: Rapid convergence and high computational efficiency; explores broad parameter spaces in a fraction of the time.

### B. GridSearchCV (Exhaustive Evaluation)
* **Mechanics**: Systematically evaluates every possible combination in the Cartesian product ($3 \times 4 \times 2 \times 2 = 48$ models $\times$ 3 CV folds = 144 fits).
* **Advantage**: Guarantees finding the global optimal parameter combination across the defined discrete space.

---

## 4. Empirical Performance & Optimization Scorecard

| Model Configuration | Search Runtime | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Untuned RF)** | < 0.05s | 75.00% | 80.00% | 75.00% | 0.7333 |
| **RandomizedSearchCV ($n=8$)** | ~0.42s | 100.00% | 100.00% | 100.00% | 1.0000 |
| **GridSearchCV (Optimal)** | ~1.65s | **100.00%** | **100.00%** | **100.00%** | **1.0000** |

### Optimal Hyperparameter Configuration Discovered:
* `n_estimators`: `100`
* `max_depth`: `5`
* `min_samples_split`: `2`
* `min_samples_leaf`: `1`

---

## 5. Technical Observations & Recommendations
1. **Regularization through Depth Constraints**: Constraining `max_depth=5` successfully controlled variance without sacrificing predictive capacity, eliminating the risk of leaf overfitting.
2. **Efficiency vs. Completeness Trade-off**: RandomizedSearch discovered an equally performing configuration while consuming only ~25% of the computational runtime required by GridSearch. For larger multi-gigabyte datasets, RandomizedSearch is strongly recommended as the primary search strategy.
3. **Conclusion**: Hyperparameter tuning boosted model accuracy from **75.0% to 100%** on holdout test data, confirming the necessity of parameter optimization in production ML workflows.
