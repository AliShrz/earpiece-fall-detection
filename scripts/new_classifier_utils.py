import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_and_analyze_models(dataset, df=None, random_state=9, test_size=0.2, 
                           imputation_strategy='mean', handle_nans='impute'):
    """
    Train final models, check overfitting, and generate confusion matrices
    
    Parameters:
    - imputation_strategy: 'mean', 'median', 'most_frequent', 'constant', 'knn', or 'drop'
    - handle_nans: 'impute', 'drop', or 'use_histgb' (use HistGradientBoostingClassifier)
    """
    if df is None:
        df = dataset.load()
    
    # Subject-based train/test split (similar to your CV approach)
    rng = np.random.default_rng(random_state)
    subjects = list(df['SubjectID'].unique())
    rng.shuffle(subjects)
    
    n_test_subjects = max(1, int(len(subjects) * test_size))
    test_subjects = subjects[:n_test_subjects]
    train_subjects = subjects[n_test_subjects:]
    
    train_df = df[df['SubjectID'].isin(train_subjects)]
    test_df = df[df['SubjectID'].isin(test_subjects)]
    
    X_train, y_train = dataset.get_X_y(train_df)
    X_test, y_test = dataset.get_X_y(test_df)
    
    print(f"Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
    print(f"Train set: X: {X_train.shape}, y: {y_train.shape} ([ADLs, Falls]: {np.bincount(y_train)})")
    print(f"Test set: X: {X_test.shape}, y: {y_test.shape} ([ADLs, Falls]: {np.bincount(y_test)})")
    
    # Check for NaN values
    nan_count_train = np.isnan(X_train).sum()
    nan_count_test = np.isnan(X_test).sum()
    print(f"NaN values in training set: {nan_count_train}")
    print(f"NaN values in test set: {nan_count_test}")
    
    # Handle NaN values
    if handle_nans == 'drop':
        print("Dropping rows with NaN values...")
        # Drop rows with NaN values
        nan_mask_train = ~np.isnan(X_train).any(axis=1)
        nan_mask_test = ~np.isnan(X_test).any(axis=1)
        X_train = X_train[nan_mask_train]
        y_train = y_train[nan_mask_train]
        X_test = X_test[nan_mask_test]
        y_test = y_test[nan_mask_test]
        print(f"After dropping NaN - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Initialize models (standard ones)
        models = {
            'ExtraTreesClassifier': ExtraTreesClassifier(random_state=random_state, n_estimators=100),
            'RandomForestClassifier': RandomForestClassifier(random_state=random_state, n_estimators=100)
        }
        
    elif handle_nans == 'use_histgb':
        print("Using HistGradientBoostingClassifier (handles NaN natively)...")
        # Use models that handle NaN natively
        models = {
            'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=random_state),
            'ExtraTreesClassifier_with_imputation': Pipeline([
                ('imputer', SimpleImputer(strategy=imputation_strategy)),
                ('classifier', ExtraTreesClassifier(random_state=random_state, n_estimators=100))
            ]),
            'RandomForestClassifier_with_imputation': Pipeline([
                ('imputer', SimpleImputer(strategy=imputation_strategy)),
                ('classifier', RandomForestClassifier(random_state=random_state, n_estimators=100))
            ])
        }
        
    else:  # handle_nans == 'impute'
        print(f"Imputing NaN values using strategy: {imputation_strategy}")
        if imputation_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=imputation_strategy)
        
        # Create pipelines with imputation
        models = {
            'ExtraTreesClassifier': Pipeline([
                ('imputer', imputer),
                ('classifier', ExtraTreesClassifier(random_state=random_state, n_estimators=100))
            ]),
            'RandomForestClassifier': Pipeline([
                ('imputer', imputer),
                ('classifier', RandomForestClassifier(random_state=random_state, n_estimators=100))
            ])
        }
    
    results = {}
    
    # Train models and collect results
    for name, model in models.items():
        print(f"\n=== {name} ===")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics for overfitting check
        train_auc = roc_auc_score(y_train, train_pred_proba)
        test_auc = roc_auc_score(y_test, test_pred_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'overfitting_gap': train_auc - test_auc
        }
        
        print(f"Train AUC: {train_auc:.3f}")
        print(f"Test AUC: {test_auc:.3f}")
        print(f"Overfitting gap: {train_auc - test_auc:.3f}")
        if train_auc - test_auc > 0.1:
            print("⚠️  Potential overfitting detected (gap > 0.1)")
        else:
            print("✅ Good generalization")
    
    return results, (X_train, y_train, X_test, y_test)

def plot_confusion_matrices(results, y_train, y_test):
    """Plot confusion matrices for train and test sets"""
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 5*n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, result) in enumerate(results.items()):
        # Training confusion matrix
        cm_train = confusion_matrix(y_train, result['train_pred'])
        sns.heatmap(cm_train, annot=True, fmt='d', ax=axes[idx, 0],
                   xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall'])
        axes[idx, 0].set_title(f'{name} - Training Set')
        axes[idx, 0].set_xlabel('Predicted')
        axes[idx, 0].set_ylabel('Actual')
        
        # Test confusion matrix
        cm_test = confusion_matrix(y_test, result['test_pred'])
        sns.heatmap(cm_test, annot=True, fmt='d', ax=axes[idx, 1],
                   xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall'])
        axes[idx, 1].set_title(f'{name} - Test Set')
        axes[idx, 1].set_xlabel('Predicted')
        axes[idx, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(models, X_train, y_train, cv=5):
    """Plot learning curves to visualize overfitting"""
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    
    if len(models) == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[idx].plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        axes[idx].plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
        axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        axes[idx].set_title(f'{name} Learning Curve')
        axes[idx].set_xlabel('Training Set Size')
        axes[idx].set_ylabel('Accuracy Score')
        axes[idx].legend(loc='best')
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(results, handle_nans='impute'):
    """Analyze and plot feature importance"""
    fig, axes = plt.subplots(1, len(results), figsize=(8*len(results), 6))
    
    if len(results) == 1:
        axes = [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        model = result['model']
        
        # Handle pipeline models vs direct models
        if hasattr(model, 'named_steps'):  # Pipeline
            classifier = model.named_steps['classifier']
        else:
            classifier = model
            
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            axes[idx].bar(range(len(indices)), importances[indices])
            axes[idx].set_title(f'{name} - Top 20 Feature Importances')
            axes[idx].set_xlabel('Feature Index')
            axes[idx].set_ylabel('Importance')
            axes[idx].set_xticks(range(len(indices)))
            axes[idx].set_xticklabels(indices, rotation=45)
        else:
            axes[idx].text(0.5, 0.5, 'Feature importance\nnot available', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{name} - Feature Importances')
    
    plt.tight_layout()
    plt.show()

# Main execution
def full_model_analysis(dataset, df=None, handle_nans='impute', imputation_strategy='mean'):
    """
    Complete analysis pipeline
    
    Parameters:
    - handle_nans: 'impute', 'drop', or 'use_histgb'
    - imputation_strategy: 'mean', 'median', 'most_frequent', 'constant', 'knn'
    """
    
    # 1. Train models and check overfitting
    print("=== TRAINING MODELS ===")
    results, (X_train, y_train, X_test, y_test) = train_and_analyze_models(
        dataset, df, handle_nans=handle_nans, imputation_strategy=imputation_strategy)
    
    # 2. Plot confusion matrices
    print("\n=== CONFUSION MATRICES ===")
    plot_confusion_matrices(results, y_train, y_test)
    
    # 3. Plot learning curves for overfitting analysis
    print("\n=== LEARNING CURVES ===")
    models_dict = {name: result['model'] for name, result in results.items()}
    plot_learning_curves(models_dict, X_train, y_train)
    
    # 4. Feature importance analysis
    print("\n=== FEATURE IMPORTANCE ===")
    analyze_feature_importance(results, handle_nans)
    
    # 5. Print detailed classification reports
    print("\n=== DETAILED CLASSIFICATION REPORTS ===")
    for name, result in results.items():
        print(f"\n{name} - Test Set Classification Report:")
        print(classification_report(y_test, result['test_pred'], target_names=['ADL', 'Fall']))
    
    return results

