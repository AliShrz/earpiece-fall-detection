# ========================================
# QUICK DOMAIN ADAPTATION TEST
# ========================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
                if 'fall' in idx.lower():
                    id_label_pairs.append((idx, 'Fall'))
                else:
                    id_label_pairs.append((idx, 'Walking'))
        
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

def load_uma_test_results():
    """Load previously saved UMA test results"""
    try:
        with open('uma_test_results.json', 'r') as f:
            uma_results = json.load(f)
        return uma_results
    except:
        return None

def domain_robust_feature_selection(X_source, X_target, feature_names, k=500):
    """
    Select features that are robust across domains using statistical similarity
    """
    print(f"\n🔍 Selecting domain-robust features...")
    
    robust_scores = []
    
    for i, feature_name in enumerate(feature_names):
        if i >= X_source.shape[1] or i >= X_target.shape[1]:
            continue
            
        source_vals = X_source.iloc[:, i]
        target_vals = X_target.iloc[:, i]
        
        # Remove NaN values
        source_vals = source_vals.dropna()
        target_vals = target_vals.dropna()
        
        if len(source_vals) == 0 or len(target_vals) == 0:
            robust_scores.append((feature_name, 0.0))
            continue
        
        try:
            # Kolmogorov-Smirnov test - higher p-value means more similar distributions
            ks_stat, ks_p_value = stats.ks_2samp(source_vals, target_vals)
            
            # Mann-Whitney U test
            mw_stat, mw_p_value = stats.mannwhitneyu(source_vals, target_vals, alternative='two-sided')
            
            # Combine similarity scores
            similarity_score = (ks_p_value + mw_p_value) / 2
            robust_scores.append((feature_name, similarity_score))
            
        except Exception as e:
            robust_scores.append((feature_name, 0.0))
    
    # Sort by similarity score (descending)
    robust_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top k features
    robust_features = [name for name, score in robust_scores[:k]]
    
    print(f"✓ Selected {len(robust_features)} domain-robust features")
    print(f"📊 Top 5 robust features:")
    for i, (name, score) in enumerate(robust_scores[:5]):
        print(f"  {i+1}. {name[:50]}...: {score:.4f}")
    
    return robust_features

def discriminative_feature_selection(X, y, method='mutual_info', k=300):
    """
    Select features that are discriminative for classification
    """
    print(f"\n🎯 Selecting discriminative features using {method}...")
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_classif, k=k)
    
    X_selected = selector.fit_transform(X, y)
    selected_mask = selector.get_support()
    selected_features = [name for name, selected in zip(X.columns, selected_mask) if selected]
    
    print(f"✓ Selected {len(selected_features)} discriminative features")
    
    return selected_features, selector

def quick_domain_adaptation_test():
    """Quick test of domain adaptation approaches"""
    print("="*80)
    print("QUICK DOMAIN ADAPTATION TEST")
    print("="*80)
    
    # Load SisFall data
    X_sisfall, y_sisfall = load_sisfall_data()
    if X_sisfall is None:
        print("❌ Failed to load SisFall data")
        return
    
    # Load UMA test results
    uma_results = load_uma_test_results()
    if uma_results is None:
        print("❌ No UMA test results found")
        return
    
    baseline_uma_accuracy = uma_results['accuracy']
    print(f"\n📊 Baseline UMA Accuracy: {baseline_uma_accuracy:.4f}")
    
    # Split SisFall data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sisfall, y_sisfall, test_size=0.2, random_state=42, stratify=y_sisfall
    )
    
    print(f"📊 Dataset sizes:")
    print(f"  SisFall Train: {X_train.shape}")
    print(f"  SisFall Test: {X_test.shape}")
    
    # Load UMA data for feature analysis (smaller sample)
    try:
        from improved_classification import load_uma_data, extract_uma_features
        
        print("\n📂 Loading small UMA sample for analysis...")
        uma_data, uma_labels, uma_ids = load_uma_data('output_uma', max_files_per_activity=25)
        
        if uma_data is not None:
            # Get training feature names
            training_feature_names = X_sisfall.columns.tolist()
            
            # Extract UMA features (smaller sample)
            uma_features = extract_uma_features(uma_data, training_feature_names)
            
            if not uma_features.empty:
                print(f"✓ UMA sample loaded: {uma_features.shape}")
                
                # Test domain adaptation approaches
                print(f"\n{'='*60}")
                print("DOMAIN ADAPTATION APPROACHES")
                print(f"{'='*60}")
                
                results = {}
                
                # Approach 1: Domain-robust feature selection
                print(f"\n🔬 Approach 1: Domain-Robust Feature Selection")
                print("-" * 50)
                
                robust_features = domain_robust_feature_selection(
                    X_train, uma_features, X_train.columns.tolist(), k=800
                )
                
                # Filter to robust features
                X_train_robust = X_train[robust_features]
                X_test_robust = X_test[robust_features]
                
                # Select discriminative features from robust ones
                discriminative_features, selector = discriminative_feature_selection(
                    X_train_robust, y_train, method='mutual_info', k=400
                )
                
                # Train model on selected features
                X_train_final = X_train_robust[discriminative_features]
                X_test_final = X_test_robust[discriminative_features]
                
                print(f"📚 Training Random Forest on {X_train_final.shape[1]} features...")
                rf_adapted = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                
                rf_adapted.fit(X_train_final, y_train)
                
                # Test on SisFall
                sisfall_score = rf_adapted.score(X_test_final, y_test)
                print(f"✓ Domain-Adapted SisFall Accuracy: {sisfall_score:.4f}")
                
                results['domain_robust'] = {
                    'sisfall_accuracy': sisfall_score,
                    'selected_features': len(discriminative_features),
                    'robust_features': len(robust_features)
                }
                
                # Approach 2: Standard feature selection (for comparison)
                print(f"\n🔬 Approach 2: Standard Feature Selection (Baseline)")
                print("-" * 50)
                
                standard_features, standard_selector = discriminative_feature_selection(
                    X_train, y_train, method='mutual_info', k=400
                )
                
                X_train_standard = X_train[standard_features]
                X_test_standard = X_test[standard_features]
                
                rf_standard = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                
                rf_standard.fit(X_train_standard, y_train)
                standard_sisfall_score = rf_standard.score(X_test_standard, y_test)
                
                print(f"✓ Standard SisFall Accuracy: {standard_sisfall_score:.4f}")
                
                results['standard'] = {
                    'sisfall_accuracy': standard_sisfall_score,
                    'selected_features': len(standard_features)
                }
                
                # Approach 3: Conservative Random Forest
                print(f"\n🔬 Approach 3: Conservative Random Forest")
                print("-" * 50)
                
                # Use fewer, more conservative features
                conservative_features, _ = discriminative_feature_selection(
                    X_train, y_train, method='f_classif', k=200
                )
                
                X_train_conservative = X_train[conservative_features]
                X_test_conservative = X_test[conservative_features]
                
                rf_conservative = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                
                rf_conservative.fit(X_train_conservative, y_train)
                conservative_sisfall_score = rf_conservative.score(X_test_conservative, y_test)
                
                print(f"✓ Conservative SisFall Accuracy: {conservative_sisfall_score:.4f}")
                
                results['conservative'] = {
                    'sisfall_accuracy': conservative_sisfall_score,
                    'selected_features': len(conservative_features)
                }
                
                # Save models for testing
                joblib.dump(rf_adapted, 'rf_domain_adapted.pkl')
                joblib.dump(rf_standard, 'rf_standard_baseline.pkl')
                joblib.dump(rf_conservative, 'rf_conservative.pkl')
                
                # Save feature lists
                with open('selected_features.json', 'w') as f:
                    json.dump({
                        'domain_robust': discriminative_features,
                        'standard': standard_features,
                        'conservative': conservative_features
                    }, f, indent=2)
                
                print(f"\n{'='*60}")
                print("RESULTS SUMMARY")
                print(f"{'='*60}")
                
                print(f"\n📊 APPROACH COMPARISON:")
                print("-" * 50)
                print(f"{'Method':<20} {'Features':<10} {'SisFall Acc':<12}")
                print("-" * 50)
                
                for method, metrics in results.items():
                    features = metrics.get('selected_features', 0)
                    sisfall_acc = metrics['sisfall_accuracy']
                    print(f"{method:<20} {features:<10} {sisfall_acc:<12.4f}")
                
                print("-" * 50)
                
                # Save results
                experiment_results = {
                    'baseline_uma_accuracy': baseline_uma_accuracy,
                    'approaches': results,
                    'dataset_info': {
                        'sisfall_train_size': X_train.shape[0],
                        'sisfall_test_size': X_test.shape[0],
                        'uma_sample_size': uma_features.shape[0]
                    }
                }
                
                with open('quick_domain_adaptation_results.json', 'w') as f:
                    json.dump(experiment_results, f, indent=2)
                
                print(f"\n💡 KEY INSIGHTS:")
                print("1. Models saved for UMA testing: rf_domain_adapted.pkl, rf_standard_baseline.pkl, rf_conservative.pkl")
                print("2. Feature selections saved in: selected_features.json")
                print("3. Results saved in: quick_domain_adaptation_results.json")
                print("4. Next step: Test these models on the full UMA dataset")
                
                return experiment_results
            
            else:
                print("❌ Failed to extract UMA features")
        else:
            print("❌ Failed to load UMA data")
    
    except Exception as e:
        print(f"❌ Error in domain adaptation test: {e}")
        import traceback
        traceback.print_exc()
        
    return None

if __name__ == "__main__":
    results = quick_domain_adaptation_test()
