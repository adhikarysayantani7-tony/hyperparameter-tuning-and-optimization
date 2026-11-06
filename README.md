# Hyperparameter Tuning and Optimization

## Project Overview
This repository implements an end-to-end hyperparameter tuning and optimization pipeline applied to a Random Forest Classifier on credit risk data. We compare an untuned baseline against RandomizedSearchCV and GridSearchCV across multiple hyperparameters, evaluate accuracy and F1 improvements, and analyze optimization runtime trade-offs.

## Repository Architecture
- README.md
- requirements.txt
- data/dataset.csv
- src/hyperparameter_tuning.py
- notebooks/hyperparameter_tuning.ipynb
- docs/tuning_report.md

## Hyperparameters Tuned
* `n_estimators`: [50, 100, 200] (Ensemble tree count)
* `max_depth`: [3, 5, 10, None] (Tree depth constraint to control variance)
* `min_samples_split`: [2, 5] (Internal split threshold)
* `min_samples_leaf`: [1, 2] (Leaf node sample threshold)

## Performance Comparison Scorecard
* Baseline (Untuned RF): Test Accuracy = 75.00%, F1-Score = 0.7333 (Runtime < 0.05s)
* RandomizedSearchCV: Test Accuracy = 100.00%, F1-Score = 1.0000 (Runtime ~0.42s)
* GridSearchCV (Optimal): Test Accuracy = 100.00%, F1-Score = 1.0000 (Runtime ~1.65s)

## Optimal Hyperparameters Discovered
* `n_estimators`: 100
* `max_depth`: 5
* `min_samples_split`: 2
* `min_samples_leaf`: 1

## Documentation & Analysis
A comprehensive technical report detailing parameter search spaces, efficiency vs. completeness trade-offs, and holdout evaluations is provided in `docs/tuning_report.md`.

## How to Run
1. Clone this repository:
   git clone https://github.com/adhikarysayantani7-tony/hyperparameter-tuning-and-optimization.git
2. Install dependencies:
   pip install -r requirements.txt
3. Run the optimization script:
   python src/hyperparameter_tuning.py
4. Open the interactive notebook:
   jupyter notebook notebooks/hyperparameter_tuning.ipynb
