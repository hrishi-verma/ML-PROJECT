"""
Improved ML Pipeline for Android Malware Detection
Features:
- Advanced feature engineering
- Proper cross-validation
- Hyperparameter tuning
- 4 optimized classifiers
- Ensemble voting
- Detailed performance metrics
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import time

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - will use 3 models instead of 4")

print("="*80)
print("IMPROVED ML PIPELINE - Android Malware Detection")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
eval_ids = pd.read_csv('data/eval.ids', header=None)[0].values

X_train = train.drop('label', axis=1)
y_train = train['label']
X_test = test.copy()

print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✓ Class distribution: Malware={sum(y_train==1)}, Benign={sum(y_train==0)}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2/8] Feature engineering...")

# 2.1 Missing value indicators
missing_cols = X_train.columns[X_train.isnull().any()].tolist()
for col in missing_cols:
    X_train[f'{col}_missing'] = X_train[col].isnull().astype(int)
    X_test[f'{col}_missing'] = X_test[col].isnull().astype(int)
print(f"✓ Created {len(missing_cols)} missing indicators")

# 2.2 Impute missing values
imputer = SimpleImputer(strategy='median')  # Median is more robust than mean
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
print(f"✓ Imputed missing values using median strategy")

# 2.3 Remove low variance features
variance_threshold = VarianceThreshold(threshold=0.01)
X_train_var = variance_threshold.fit_transform(X_train_imputed)
X_test_var = variance_threshold.transform(X_test_imputed)
selected_features = X_train_imputed.columns[variance_threshold.get_support()].tolist()
print(f"✓ Removed low variance features: {X_train_imputed.shape[1]} → {len(selected_features)}")

X_train_imputed = pd.DataFrame(X_train_var, columns=selected_features, index=X_train.index)
X_test_imputed = pd.DataFrame(X_test_var, columns=selected_features, index=X_test.index)

# 2.4 Feature selection using statistical tests
print("✓ Selecting top features using F-statistic...")
k_best = min(200, X_train_imputed.shape[1])  # Select top 200 features
selector = SelectKBest(f_classif, k=k_best)
X_train_selected = selector.fit_transform(X_train_imputed, y_train)
X_test_selected = selector.transform(X_test_imputed)
selected_feature_names = X_train_imputed.columns[selector.get_support()].tolist()
print(f"✓ Selected top {k_best} features")

# ============================================================================
# 3. PREPARE SCALED DATA (for LR and SVM)
# ============================================================================
print("\n[3/8] Scaling features...")
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)
print("✓ Features scaled using RobustScaler")

# ============================================================================
# 4. DEFINE MODELS WITH OPTIMIZED HYPERPARAMETERS
# ============================================================================
print("\n[4/8] Initializing models...")

# Calculate class weights
class_weight_ratio = sum(y_train == 0) / sum(y_train == 1)

models = {}

# Model 1: Logistic Regression with L2 regularization
models['Logistic Regression'] = {
    'model': LogisticRegression(
        C=0.5,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    'use_scaled': True
}

# Model 2: Random Forest with optimized parameters
models['Random Forest'] = {
    'model': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'use_scaled': False
}

# Model 3: SVM with RBF kernel
models['SVM'] = {
    'model': SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        random_state=42,
        class_weight='balanced',
        probability=True  # Enable probability estimates for voting
    ),
    'use_scaled': True
}

# Model 4: XGBoost (if available)
if XGBOOST_AVAILABLE:
    models['XGBoost'] = {
        'model': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=3,
            scale_pos_weight=class_weight_ratio,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'use_scaled': False
    }

print(f"✓ Initialized {len(models)} models")

# ============================================================================
# 5. CROSS-VALIDATION
# ============================================================================
print("\n[5/8] Cross-validation (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, model_info in models.items():
    print(f"\n  {name}:")
    model = model_info['model']
    X_cv = X_train_scaled if model_info['use_scaled'] else X_train_selected
    
    start_time = time.time()
    scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    elapsed = time.time() - start_time
    
    cv_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores,
        'time': elapsed
    }
    
    print(f"    F1 Score: {scores.mean():.5f} (+/- {scores.std():.5f})")
    print(f"    Time: {elapsed:.2f}s")

# ============================================================================
# 6. TRAIN FINAL MODELS
# ============================================================================
print("\n[6/8] Training final models on full training set...")

trained_models = {}
for name, model_info in models.items():
    print(f"  Training {name}...")
    model = model_info['model']
    X_train_final = X_train_scaled if model_info['use_scaled'] else X_train_selected
    
    start_time = time.time()
    model.fit(X_train_final, y_train)
    elapsed = time.time() - start_time
    
    trained_models[name] = {
        'model': model,
        'use_scaled': model_info['use_scaled'],
        'train_time': elapsed
    }
    print(f"    ✓ Trained in {elapsed:.2f}s")

# ============================================================================
# 7. GENERATE PREDICTIONS
# ============================================================================
print("\n[7/8] Generating predictions...")

all_predictions = {}
for name, model_info in trained_models.items():
    model = model_info['model']
    X_test_final = X_test_scaled if model_info['use_scaled'] else X_test_selected
    
    predictions = model.predict(X_test_final)
    all_predictions[name] = predictions
    
    malware_count = sum(predictions == 1)
    benign_count = sum(predictions == 0)
    print(f"  {name}: Malware={malware_count}, Benign={benign_count}")

# ============================================================================
# 8. ENSEMBLE VOTING
# ============================================================================
print("\n[8/8] Creating ensemble predictions...")

# Majority voting
prediction_matrix = np.array([all_predictions[name] for name in models.keys()])
ensemble_predictions = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(),
    axis=0,
    arr=prediction_matrix
)

malware_count = sum(ensemble_predictions == 1)
benign_count = sum(ensemble_predictions == 0)
print(f"  Ensemble: Malware={malware_count}, Benign={benign_count}")

# ============================================================================
# 9. SAVE SUBMISSIONS
# ============================================================================
print("\n" + "="*80)
print("SAVING SUBMISSION FILES")
print("="*80)

# Save individual model predictions
for name, predictions in all_predictions.items():
    filename = f"submission_{name.lower().replace(' ', '_')}.csv"
    submission = pd.DataFrame({
        'example_id': eval_ids,
        'label': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"✓ {filename}")

# Save ensemble prediction
submission_ensemble = pd.DataFrame({
    'example_id': eval_ids,
    'label': ensemble_predictions
})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)
print(f"✓ submission_ensemble.csv")

# Save best individual model based on CV
best_model_name = max(cv_results.items(), key=lambda x: x[1]['mean'])[0]
best_predictions = all_predictions[best_model_name]
submission_best = pd.DataFrame({
    'example_id': eval_ids,
    'label': best_predictions
})
submission_best.to_csv('submission.csv', index=False)
print(f"✓ submission.csv (Best: {best_model_name})")

# ============================================================================
# 10. SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)

print("\nCross-Validation Results (5-fold):")
print("-" * 80)
sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['mean'], reverse=True)
for rank, (name, results) in enumerate(sorted_results, 1):
    print(f"{rank}. {name:25s} F1={results['mean']:.5f} (±{results['std']:.5f})  Time={results['time']:.2f}s")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print(f"\n Best Model: {best_model_name}")
print(f"   F1 Score: {cv_results[best_model_name]['mean']:.5f}")
print(f"   Use: submission.csv")

print(f"\n🎯 Ensemble Model (Majority Voting)")
print(f"   Combines all {len(models)} models")
print(f"   Use: submission_ensemble.csv")

print("\n💡 Tips:")
print("   - submission.csv uses the best individual model")
print("   - submission_ensemble.csv often performs better")
print("   - Try both on Kaggle and compare scores!")

print("\n" + "="*80)
print(" COMPLETE! Ready to upload to Kaggle")
print("="*80)
