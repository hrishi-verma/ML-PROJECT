# Android Malware Detection - ML Pipeline

A machine learning pipeline for detecting Android malware using ensemble methods and advanced feature engineering.

## Overview

This project implements an improved ML pipeline that trains 4 classifiers and combines them using ensemble voting to achieve better performance than individual models.

## Features

- Advanced feature engineering with missing value indicators
- Statistical feature selection (top 200 features)
- 4 optimized classifiers: Logistic Regression, Random Forest, SVM, XGBoost
- 5-fold stratified cross-validation
- Ensemble predictions using majority voting
- Generates 6 submission files for comparison

## Requirements

```
pandas
numpy
scikit-learn
xgboost (optional)
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost
```

## Usage

### Quick Start

```bash
python3 improved_ml_pipeline.py
```

### Expected Output

The script generates 6 CSV files:

1. `submission.csv` - Best individual model (automatically selected)
2. `submission_ensemble.csv` - Ensemble predictions (recommended)
3. `submission_logistic_regression.csv` - Logistic Regression predictions
4. `submission_random_forest.csv` - Random Forest predictions
5. `submission_svm.csv` - SVM predictions
6. `submission_xgboost.csv` - XGBoost predictions

## Data Structure

Place your data files in a `data/` directory:

```
data/
├── train.csv      # Training data with 'label' column
├── test.csv       # Test data without labels
└── eval.ids       # Test sample IDs
```

## Pipeline Steps

1. **Data Loading** - Loads training and test datasets
2. **Feature Engineering** - Creates missing indicators, removes low variance features
3. **Feature Selection** - Selects top 200 features using F-statistic
4. **Preprocessing** - Applies RobustScaler for scaling
5. **Cross-Validation** - 5-fold stratified CV to evaluate models
6. **Training** - Trains all 4 models on full training set
7. **Prediction** - Generates predictions for test set
8. **Ensemble** - Combines predictions using majority voting

## Model Performance

Based on 5-fold cross-validation:

| Model | F1 Score | Std Dev |
|-------|----------|---------|
| SVM | 0.96644 | 0.00348 |
| Logistic Regression | 0.96156 | 0.00324 |
| Random Forest | 0.96054 | 0.00649 |
| XGBoost | 0.95526 | 0.00410 |

The ensemble typically achieves F1 scores around 0.967 or higher.

## Which Submission File to Use

**Recommended:** Start with `submission_ensemble.csv` as ensemble methods typically perform better on unseen data.

**Alternative:** Use `submission.csv` which contains predictions from the best individual model based on cross-validation scores.

## Technical Details

### Feature Engineering
- Missing value indicators for all features with missing data
- Median imputation (more robust than mean)
- Variance threshold filtering (removes features with <1% variance)
- SelectKBest with F-statistic (selects top 200 features)

### Preprocessing
- RobustScaler for feature scaling (more robust to outliers than StandardScaler)
- Separate scaling for models that require it (Logistic Regression, SVM)

### Models
All models use balanced class weights to handle class imbalance.

**Logistic Regression:**
- C=0.5, L2 penalty, liblinear solver

**Random Forest:**
- 200 estimators, max_depth=15, sqrt features

**SVM:**
- RBF kernel, C=1.0, probability estimates enabled

**XGBoost:**
- 200 estimators, max_depth=6, learning_rate=0.05
- Auto-calculated scale_pos_weight

### Ensemble Method
- Type: Hard voting (majority)
- Combines predictions from all 4 models
- Final prediction is the mode of individual predictions

## Output Format

All submission files follow the same format:

```csv
example_id,label
0,1
1,1
2,0
...
```

Where label is: 0 = Benign, 1 = Malware

## Notes

- Runtime is approximately 10 seconds on a modern laptop
- Memory usage is around 2GB RAM

## License

This project is for educational purposes.

## Author

Created for the Fall 2025 ML Course - Android Malware Detection Competition
