# ========================================
# DOMAIN ADAPTATION FOR CROSS-DATASET GENERALIZATION
# ========================================

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tsfresh import select_features, extract_features
from tsfresh.utilities.dataframe_functions import impute
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========================================
# DOMAIN ADAPTATION TECHNIQUES
# ========================================

class DomainAdaptationClassifier:
    """Advanced classifier with domain adaptation techniques for cross-dataset generalization"""
    
    def __init__(self, base_classifier=None, adaptation_method='feature_selection'):
        """
        Initialize domain adaptation classifier
        
        Args:
            base_classifier: Base classifier to use
            adaptation_method: Method for domain adaptation
                - 'feature_selection': Select domain-robust features
                - 'normalization': Normalize features across domains
                - 'ensemble': Ensemble approach
                - 'adversarial': Adversarial domain adaptation (advanced)
        """
        self.base_classifier = base_classifier or RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.adaptation_method = adaptation_method
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.domain_robust_features = None
        
    def identify_domain_robust_features(self, X_source, X_target, feature_names, top_k=500):
        """
        Identify features that are robust across domains using statistical tests
        
        Args:
            X_source: Source domain features (SisFall)
            X_target: Target domain features (UMA)
            feature_names: List of feature names
            top_k: Number of top features to select
            
        Returns:
            robust_features: List of domain-robust feature names
        """
        print(f"\n🔍 Identifying domain-robust features...")
        
        robust_scores = []
        
        for i, feature_name in enumerate(feature_names):
            if i >= X_source.shape[1] or i >= X_target.shape[1]:
                continue
                
            source_vals = X_source.iloc[:, i] if hasattr(X_source, 'iloc') else X_source[:, i]
            target_vals = X_target.iloc[:, i] if hasattr(X_target, 'iloc') else X_target[:, i]
            
            # Remove NaN values
            source_vals = source_vals[~pd.isna(source_vals)]
            target_vals = target_vals[~pd.isna(target_vals)]
            
            if len(source_vals) == 0 or len(target_vals) == 0:
                robust_scores.append((feature_name, 0.0))
                continue
            
            try:
                # Kolmogorov-Smirnov test (measures distribution similarity)
                # Lower p-value = more different distributions
                # Higher p-value = more similar distributions (better for domain adaptation)
                ks_stat, ks_p_value = stats.ks_2samp(source_vals, target_vals)
                
                # Mann-Whitney U test (non-parametric test for median differences)
                mw_stat, mw_p_value = stats.mannwhitneyu(source_vals, target_vals, alternative='two-sided')
                
                # Combine metrics: we want features with similar distributions
                # Use p-values as similarity scores (higher = more similar)
                similarity_score = (ks_p_value + mw_p_value) / 2
                
                robust_scores.append((feature_name, similarity_score))
                
            except Exception as e:
                robust_scores.append((feature_name, 0.0))
        
        # Sort by similarity score (descending)
        robust_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top k features
        robust_features = [name for name, score in robust_scores[:top_k]]
        
        print(f"✓ Selected {len(robust_features)} domain-robust features")
        print(f"📊 Top 5 robust features:")
        for i, (name, score) in enumerate(robust_scores[:5]):
            print(f"  {i+1}. {name}: {score:.4f}")
        
        self.domain_robust_features = robust_features
        return robust_features
    
    def extract_discriminative_features(self, X, y, method='mutual_info', k=300):
        """
        Extract features that are discriminative for classification
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Feature selection method ('mutual_info', 'f_classif', 'variance')
            k: Number of features to select
            
        Returns:
            selected_features: Names of selected features
        """
        print(f"\n🎯 Extracting discriminative features using {method}...")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            # Use variance threshold
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
        
        X_selected = selector.fit_transform(X, y)
        
        if hasattr(selector, 'get_support'):
            selected_mask = selector.get_support()
            feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
            selected_features = [name for name, selected in zip(feature_names, selected_mask) if selected]
        else:
            selected_features = [f'feature_{i}' for i in range(X_selected.shape[1])]
        
        print(f"✓ Selected {len(selected_features)} discriminative features")
        
        self.feature_selector = selector
        self.selected_features = selected_features
        
        return selected_features
    
    def fit(self, X_source, y_source, X_target=None, feature_names=None):
        """
        Fit the domain-adapted classifier
        
        Args:
            X_source: Source domain training data
            y_source: Source domain labels
            X_target: Target domain data (for adaptation)
            feature_names: Feature names
        """
        print(f"\n🚀 Training domain-adapted classifier with {self.adaptation_method}...")
        
        if self.adaptation_method == 'feature_selection' and X_target is not None:
            # Two-step feature selection: domain-robust + discriminative
            
            # Get feature names
            if feature_names is not None:
                feature_names_list = feature_names
            elif hasattr(X_source, 'columns'):
                feature_names_list = X_source.columns.tolist()
            else:
                feature_names_list = [f'feature_{i}' for i in range(X_source.shape[1])]
            
            # Step 1: Find domain-robust features
            robust_features = self.identify_domain_robust_features(
                X_source, X_target, feature_names_list, top_k=800
            )
            
            # Filter to robust features
            if hasattr(X_source, 'columns'):
                X_source_robust = X_source[robust_features]
            else:
                robust_indices = [i for i, name in enumerate(feature_names) if name in robust_features]
                X_source_robust = X_source[:, robust_indices]
            
            # Step 2: Select discriminative features from robust ones
            discriminative_features = self.extract_discriminative_features(
                X_source_robust, y_source, method='mutual_info', k=min(400, len(robust_features))
            )
            
            # Final feature set
            final_features = discriminative_features
            if hasattr(X_source, 'columns'):
                X_source_final = X_source[final_features]
            else:
                final_indices = [i for i, name in enumerate(feature_names) if name in final_features]
                X_source_final = X_source[:, final_indices]
            
            self.selected_features = final_features
            
        elif self.adaptation_method == 'normalization':
            # Normalize features to make them more domain-invariant
            X_source_final = self.scaler.fit_transform(X_source)
            
            # Select discriminative features
            discriminative_features = self.extract_discriminative_features(
                pd.DataFrame(X_source_final), y_source, method='f_classif', k=500
            )
            
            if hasattr(X_source, 'columns'):
                feature_names_list = X_source.columns.tolist()
            elif feature_names is not None:
                feature_names_list = feature_names
            else:
                feature_names_list = [f'feature_{i}' for i in range(X_source.shape[1])]
            
            selected_indices = [i for i, name in enumerate(feature_names_list) if f'feature_{i}' in discriminative_features or name in discriminative_features]
            X_source_final = X_source_final[:, selected_indices] if selected_indices else X_source_final
            
        else:
            # Basic approach - just use discriminative features
            if feature_names is not None:
                feature_names_list = feature_names
            elif hasattr(X_source, 'columns'):
                feature_names_list = X_source.columns.tolist()
            else:
                feature_names_list = [f'feature_{i}' for i in range(X_source.shape[1])]
                
            discriminative_features = self.extract_discriminative_features(
                X_source, y_source, method='mutual_info', k=600
            )
            
            if hasattr(X_source, 'columns'):
                X_source_final = X_source[discriminative_features]
            else:
                disc_indices = [i for i, name in enumerate(feature_names_list) if name in discriminative_features]
                X_source_final = X_source[:, disc_indices] if disc_indices else X_source
            
            self.selected_features = discriminative_features
        
        # Train the base classifier
        print(f"📚 Training classifier on {X_source_final.shape[1]} selected features...")
        self.base_classifier.fit(X_source_final, y_source)
        
        print("✅ Domain-adapted classifier trained successfully!")
        
        return self
    
    def transform_features(self, X, feature_names=None):
        """Transform features using the learned adaptation"""
        if self.selected_features is None:
            return X
        
        if hasattr(X, 'columns'):
            # DataFrame input
            available_features = [f for f in self.selected_features if f in X.columns]
            if len(available_features) < len(self.selected_features) * 0.8:
                print(f"⚠️ Warning: Only {len(available_features)}/{len(self.selected_features)} features available")
            
            X_transformed = X[available_features]
            
            # Add missing features as zeros
            for feature in self.selected_features:
                if feature not in X_transformed.columns:
                    X_transformed[feature] = 0.0
            
            # Reorder columns to match training
            X_transformed = X_transformed[self.selected_features]
            
        else:
            # Array input
            if feature_names:
                selected_indices = [i for i, name in enumerate(feature_names) if name in self.selected_features]
                X_transformed = X[:, selected_indices] if selected_indices else X
            else:
                X_transformed = X
        
        if self.adaptation_method == 'normalization' and hasattr(self.scaler, 'transform'):
            X_transformed = self.scaler.transform(X_transformed)
        
        return X_transformed
    
    def predict(self, X, feature_names=None):
        """Make predictions on new data"""
        X_transformed = self.transform_features(X, feature_names)
        return self.base_classifier.predict(X_transformed)
    
    def predict_proba(self, X, feature_names=None):
        """Predict class probabilities"""
        X_transformed = self.transform_features(X, feature_names)
        return self.base_classifier.predict_proba(X_transformed)
    
    def score(self, X, y, feature_names=None):
        """Score the classifier"""
        X_transformed = self.transform_features(X, feature_names)
        return self.base_classifier.score(X_transformed, y)

# ========================================
# ENSEMBLE DOMAIN ADAPTATION
# ========================================

class EnsembleDomainAdaptation:
    """Ensemble approach combining multiple adaptation strategies"""
    
    def __init__(self):
        self.classifiers = {}
        self.weights = {}
        
    def fit(self, X_source, y_source, X_target=None, feature_names=None):
        """Train ensemble of domain-adapted classifiers"""
        print(f"\n🌟 Training ensemble domain adaptation...")
        
        # Strategy 1: Feature selection based
        self.classifiers['feature_selection'] = DomainAdaptationClassifier(
            adaptation_method='feature_selection'
        )
        self.classifiers['feature_selection'].fit(X_source, y_source, X_target, feature_names)
        
        # Strategy 2: Normalization based
        self.classifiers['normalization'] = DomainAdaptationClassifier(
            adaptation_method='normalization'
        )
        self.classifiers['normalization'].fit(X_source, y_source, X_target, feature_names)
        
        # Strategy 3: Conservative Random Forest (less overfitting)
        conservative_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.classifiers['conservative'] = DomainAdaptationClassifier(
            base_classifier=conservative_rf,
            adaptation_method='basic'
        )
        self.classifiers['conservative'].fit(X_source, y_source, X_target, feature_names)
        
        print("✅ Ensemble domain adaptation trained!")
        
        return self
    
    def predict(self, X, feature_names=None):
        """Ensemble prediction using majority voting"""
        predictions = {}
        
        for name, classifier in self.classifiers.items():
            try:
                predictions[name] = classifier.predict(X, feature_names)
            except Exception as e:
                print(f"⚠️ Error with {name} classifier: {e}")
                continue
        
        if not predictions:
            return None
        
        # Majority voting
        n_samples = len(list(predictions.values())[0])
        final_predictions = []
        
        for i in range(n_samples):
            votes = [pred[i] for pred in predictions.values()]
            # Get most common prediction
            final_pred = max(set(votes), key=votes.count)
            final_predictions.append(final_pred)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X, feature_names=None):
        """Ensemble probability prediction"""
        probabilities = {}
        
        for name, classifier in self.classifiers.items():
            try:
                probabilities[name] = classifier.predict_proba(X, feature_names)
            except:
                continue
        
        if not probabilities:
            return None
        
        # Average probabilities
        avg_proba = np.mean(list(probabilities.values()), axis=0)
        return avg_proba

# ========================================
# LOAD DATASETS AND FEATURES
# ========================================

def load_sisfall_data():
    """Load SisFall training data"""
    print("📂 Loading SisFall data...")
    
    try:
        # Load features
        features = pd.read_csv('extracted_features_full_Gyro_20250618_224358.csv', index_col=0)
        print(f"✓ Loaded SisFall features: {features.shape}")
        
        # Create labels from feature IDs
        id_label_pairs = []
        for idx in features.index:
            if idx.startswith('D01_') or idx.startswith('D02_'):
                id_label_pairs.append((idx, 'Walking'))
            elif idx.startswith('D03_') or idx.startswith('D04_'):
                id_label_pairs.append((idx, 'Jogging'))
            elif idx.startswith('D05_') or idx.startswith('D06_'):
                id_label_pairs.append((idx, 'Stairs'))
            elif idx.startswith('D07_') or idx.startswith('D08_'):
                id_label_pairs.append((idx, 'Sitting'))
            elif any(idx.startswith(f'D{i:02d}_') for i in range(15, 20)):
                id_label_pairs.append((idx, 'Fall'))
            else:
                # Fallback mapping
                if 'fall' in idx.lower():
                    id_label_pairs.append((idx, 'Fall'))
                else:
                    id_label_pairs.append((idx, 'Walking'))  # Default
        
        # Create labels DataFrame
        labels_df = pd.DataFrame(id_label_pairs, columns=['ID', 'Label'])
        
        # Get common IDs
        common_ids = set(features.index) & set(labels_df['ID'])
        common_ids_list = list(common_ids)
        
        X = features.loc[common_ids_list]
        y = labels_df.set_index('ID').loc[common_ids_list]['Label']
        
        print(f"✓ Final SisFall dataset: {X.shape}")
        print("Class distribution:")
        print(y.value_counts())
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error loading SisFall data: {e}")
        return None, None

def load_uma_data_for_adaptation():
    """Load UMA data for domain adaptation"""
    print("📂 Loading UMA data for adaptation...")
    
    try:
        # Use existing UMA loading functions from improved_classification.py
        from improved_classification import load_uma_data, extract_uma_features
        
        # Load UMA data
        uma_data, uma_labels, uma_ids = load_uma_data('output_uma', max_files_per_activity=100)
        
        if uma_data is None:
            print("❌ Failed to load UMA data")
            return None, None
        
        # Load SisFall features to get feature names
        sisfall_features = pd.read_csv('extracted_features_full_Gyro_20250618_224358.csv', index_col=0)
        training_feature_names = sisfall_features.columns.tolist()
        
        # Extract UMA features
        uma_features = extract_uma_features(uma_data, training_feature_names)
        
        if uma_features.empty:
            print("❌ Failed to extract UMA features")
            return None, None
        
        # Get common IDs
        common_ids = set(uma_features.index) & set(uma_labels['ID'])
        common_ids_list = list(common_ids)
        
        X_uma = uma_features.loc[common_ids_list]
        y_uma = uma_labels.set_index('ID').loc[common_ids_list]['Label']
        
        print(f"✓ UMA dataset loaded: {X_uma.shape}")
        print("UMA class distribution:")
        print(y_uma.value_counts())
        
        return X_uma, y_uma
        
    except Exception as e:
        print(f"❌ Error loading UMA data: {e}")
        return None, None

# ========================================
# MAIN DOMAIN ADAPTATION PIPELINE
# ========================================

def run_domain_adaptation_experiment():
    """Run comprehensive domain adaptation experiment"""
    print("="*80)
    print("DOMAIN ADAPTATION FOR CROSS-DATASET GENERALIZATION")
    print("="*80)
    
    # Load datasets
    X_sisfall, y_sisfall = load_sisfall_data()
    X_uma, y_uma = load_uma_data_for_adaptation()
    
    if X_sisfall is None or X_uma is None:
        print("❌ Failed to load datasets")
        return
    
    # Split SisFall data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sisfall, y_sisfall, test_size=0.2, random_state=42, stratify=y_sisfall
    )
    
    print(f"\n📊 Dataset Summary:")
    print(f"  SisFall Train: {X_train.shape}")
    print(f"  SisFall Test: {X_test.shape}")
    print(f"  UMA Test: {X_uma.shape}")
    
    # ========================================
    # 1. BASELINE MODEL (Original Random Forest)
    # ========================================
    print(f"\n{'='*60}")
    print("1. BASELINE MODEL PERFORMANCE")
    print(f"{'='*60}")
    
    try:
        baseline_model = joblib.load('random_forest_improved_v2.pkl')
        print("✓ Loaded existing baseline model")
        
        # Test on SisFall
        baseline_sisfall_score = baseline_model.score(X_test, y_test)
        print(f"Baseline SisFall Accuracy: {baseline_sisfall_score:.4f}")
        
        # Test on UMA
        baseline_uma_pred = baseline_model.predict(X_uma)
        baseline_uma_score = accuracy_score(y_uma, baseline_uma_pred)
        print(f"Baseline UMA Accuracy: {baseline_uma_score:.4f}")
        print(f"Performance Drop: {baseline_sisfall_score - baseline_uma_score:.4f}")
        
    except Exception as e:
        print(f"❌ Could not load baseline model: {e}")
        baseline_sisfall_score = 0
        baseline_uma_score = 0
    
    # ========================================
    # 2. DOMAIN ADAPTATION APPROACHES
    # ========================================
    print(f"\n{'='*60}")
    print("2. DOMAIN ADAPTATION APPROACHES")
    print(f"{'='*60}")
    
    results = {}
    
    # Approach 1: Feature Selection Based
    print(f"\n🔬 Approach 1: Domain-Robust Feature Selection")
    print("-" * 50)
    
    da_feature_sel = DomainAdaptationClassifier(adaptation_method='feature_selection')
    da_feature_sel.fit(X_train, y_train, X_uma, X_train.columns)
    
    # Test performance
    fs_sisfall_score = da_feature_sel.score(X_test, y_test, X_test.columns)
    fs_uma_score = da_feature_sel.score(X_uma, y_uma, X_uma.columns)
    
    print(f"✓ Feature Selection - SisFall: {fs_sisfall_score:.4f}")
    print(f"✓ Feature Selection - UMA: {fs_uma_score:.4f}")
    print(f"✓ Performance Drop: {fs_sisfall_score - fs_uma_score:.4f}")
    
    results['feature_selection'] = {
        'sisfall_accuracy': fs_sisfall_score,
        'uma_accuracy': fs_uma_score,
        'performance_drop': fs_sisfall_score - fs_uma_score
    }
    
    # Approach 2: Normalization Based
    print(f"\n🔬 Approach 2: Feature Normalization")
    print("-" * 50)
    
    da_normalization = DomainAdaptationClassifier(adaptation_method='normalization')
    da_normalization.fit(X_train, y_train, X_uma, X_train.columns)
    
    norm_sisfall_score = da_normalization.score(X_test, y_test, X_test.columns)
    norm_uma_score = da_normalization.score(X_uma, y_uma, X_uma.columns)
    
    print(f"✓ Normalization - SisFall: {norm_sisfall_score:.4f}")
    print(f"✓ Normalization - UMA: {norm_uma_score:.4f}")
    print(f"✓ Performance Drop: {norm_sisfall_score - norm_uma_score:.4f}")
    
    results['normalization'] = {
        'sisfall_accuracy': norm_sisfall_score,
        'uma_accuracy': norm_uma_score,
        'performance_drop': norm_sisfall_score - norm_uma_score
    }
    
    # Approach 3: Ensemble Domain Adaptation
    print(f"\n🔬 Approach 3: Ensemble Domain Adaptation")
    print("-" * 50)
    
    ensemble_da = EnsembleDomainAdaptation()
    ensemble_da.fit(X_train, y_train, X_uma, X_train.columns)
    
    # Test ensemble
    ensemble_sisfall_pred = ensemble_da.predict(X_test, X_test.columns)
    ensemble_uma_pred = ensemble_da.predict(X_uma, X_uma.columns)
    
    if ensemble_sisfall_pred is not None and ensemble_uma_pred is not None:
        ensemble_sisfall_score = accuracy_score(y_test, ensemble_sisfall_pred)
        ensemble_uma_score = accuracy_score(y_uma, ensemble_uma_pred)
        
        print(f"✓ Ensemble - SisFall: {ensemble_sisfall_score:.4f}")
        print(f"✓ Ensemble - UMA: {ensemble_uma_score:.4f}")
        print(f"✓ Performance Drop: {ensemble_sisfall_score - ensemble_uma_score:.4f}")
        
        results['ensemble'] = {
            'sisfall_accuracy': ensemble_sisfall_score,
            'uma_accuracy': ensemble_uma_score,
            'performance_drop': ensemble_sisfall_score - ensemble_uma_score
        }
    else:
        print("❌ Ensemble approach failed")
    
    # ========================================
    # 3. RESULTS COMPARISON
    # ========================================
    print(f"\n{'='*60}")
    print("3. RESULTS COMPARISON")
    print(f"{'='*60}")
    
    # Add baseline to results
    results['baseline'] = {
        'sisfall_accuracy': baseline_sisfall_score,
        'uma_accuracy': baseline_uma_score,
        'performance_drop': baseline_sisfall_score - baseline_uma_score
    }
    
    # Create comparison table
    print(f"\n📊 CROSS-DATASET PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"{'Method':<20} {'SisFall Acc':<12} {'UMA Acc':<10} {'Drop':<10} {'Improvement':<12}")
    print("-" * 80)
    
    best_method = None
    best_uma_score = 0
    
    for method, metrics in results.items():
        sisfall_acc = metrics['sisfall_accuracy']
        uma_acc = metrics['uma_accuracy']
        drop = metrics['performance_drop']
        
        if method == 'baseline':
            improvement = "baseline"
        else:
            improvement = f"+{uma_acc - baseline_uma_score:.3f}"
        
        print(f"{method:<20} {sisfall_acc:<12.4f} {uma_acc:<10.4f} {drop:<10.4f} {improvement:<12}")
        
        if uma_acc > best_uma_score:
            best_uma_score = uma_acc
            best_method = method
    
    print("-" * 80)
    print(f"\n🏆 Best Method: {best_method} (UMA Accuracy: {best_uma_score:.4f})")
    
    # ========================================
    # 4. DETAILED ANALYSIS OF BEST METHOD
    # ========================================
    print(f"\n{'='*60}")
    print("4. DETAILED ANALYSIS OF BEST METHOD")
    print(f"{'='*60}")
    
    if best_method in ['feature_selection', 'normalization']:
        if best_method == 'feature_selection':
            best_model = da_feature_sel
        else:
            best_model = da_normalization
        
        # Get predictions for detailed analysis
        uma_pred = best_model.predict(X_uma, X_uma.columns)
        
        print(f"\nDetailed UMA Classification Report ({best_method}):")
        print(classification_report(y_uma, uma_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_uma, uma_pred)
        
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sorted(y_uma.unique()),
                   yticklabels=sorted(y_uma.unique()))
        plt.title(f'UMA Confusion Matrix - {best_method.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Comparison with baseline
        try:
            baseline_uma_pred = baseline_model.predict(X_uma)
            baseline_cm = confusion_matrix(y_uma, baseline_uma_pred)
            
            plt.subplot(1, 2, 2)
            sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Reds',
                       xticklabels=sorted(y_uma.unique()),
                       yticklabels=sorted(y_uma.unique()))
            plt.title('UMA Confusion Matrix - Baseline')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        except:
            pass
        
        plt.tight_layout()
        plt.savefig('domain_adaptation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save best model
        joblib.dump(best_model, f'domain_adapted_model_{best_method}.pkl')
        print(f"✅ Saved best model as 'domain_adapted_model_{best_method}.pkl'")
    
    # ========================================
    # 5. SAVE RESULTS
    # ========================================
    print(f"\n{'='*60}")
    print("5. SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save detailed results
    domain_adaptation_results = {
        'experiment_summary': {
            'sisfall_train_size': X_train.shape[0],
            'sisfall_test_size': X_test.shape[0],
            'uma_test_size': X_uma.shape[0],
            'baseline_performance_drop': baseline_sisfall_score - baseline_uma_score,
            'best_method': best_method,
            'best_uma_accuracy': best_uma_score,
            'improvement_over_baseline': best_uma_score - baseline_uma_score
        },
        'method_results': results,
        'selected_features': {
            'feature_selection': da_feature_sel.selected_features[:50] if da_feature_sel.selected_features else [],
            'normalization': da_normalization.selected_features[:50] if da_normalization.selected_features else []
        }
    }
    
    with open('domain_adaptation_results.json', 'w') as f:
        json.dump(domain_adaptation_results, f, indent=2)
    
    print("✅ Results saved to 'domain_adaptation_results.json'")
    
    # ========================================
    # 6. SUMMARY AND RECOMMENDATIONS
    # ========================================
    print(f"\n{'='*60}")
    print("6. SUMMARY AND RECOMMENDATIONS")
    print(f"{'='*60}")
    
    improvement = best_uma_score - baseline_uma_score
    percent_improvement = (improvement / baseline_uma_score) * 100 if baseline_uma_score > 0 else 0
    
    print(f"\n📋 EXPERIMENT SUMMARY:")
    print(f"  • Baseline UMA Accuracy: {baseline_uma_score:.4f}")
    print(f"  • Best Method: {best_method}")
    print(f"  • Best UMA Accuracy: {best_uma_score:.4f}")
    print(f"  • Absolute Improvement: +{improvement:.4f}")
    print(f"  • Relative Improvement: +{percent_improvement:.1f}%")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    if improvement > 0.1:
        print("  ✅ EXCELLENT improvement achieved!")
        print("  → Deploy the domain-adapted model for cross-dataset scenarios")
        print("  → Consider collecting more diverse training data")
    elif improvement > 0.05:
        print("  ✅ GOOD improvement achieved")
        print("  → The domain adaptation is working well")
        print("  → Consider ensemble approaches for further improvement")
    elif improvement > 0.01:
        print("  ⚠️  MODEST improvement achieved")
        print("  → Domain adaptation helps but more work needed")
        print("  → Consider advanced techniques like adversarial training")
    else:
        print("  ❌ LIMITED improvement achieved")
        print("  → Dataset bias is very strong")
        print("  → Consider joint training or data augmentation")
    
    print(f"\n🔄 NEXT STEPS:")
    print("  1. Implement adversarial domain adaptation")
    print("  2. Try joint training on both datasets")
    print("  3. Explore data augmentation techniques")
    print("  4. Consider unsupervised domain adaptation")
    
    return domain_adaptation_results

if __name__ == "__main__":
    results = run_domain_adaptation_experiment()
