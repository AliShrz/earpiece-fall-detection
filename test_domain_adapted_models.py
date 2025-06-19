# ========================================
# TEST DOMAIN ADAPTED MODELS ON UMA
# ========================================

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def load_uma_test_data():
    """Load UMA test data using the same process as before"""
    try:
        from improved_classification import load_uma_data, extract_uma_features
        
        print("📂 Loading full UMA dataset for testing...")
        uma_data, uma_labels, uma_ids = load_uma_data('output_uma', max_files_per_activity=None)
        
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
        
        print(f"✓ UMA test dataset loaded: {X_uma.shape}")
        print("UMA class distribution:")
        print(y_uma.value_counts())
        
        return X_uma, y_uma
        
    except Exception as e:
        print(f"❌ Error loading UMA data: {e}")
        return None, None

def test_domain_adapted_models():
    """Test all domain-adapted models on UMA dataset"""
    print("="*80)
    print("TESTING DOMAIN ADAPTED MODELS ON UMA DATASET")
    print("="*80)
    
    # Load UMA test data
    X_uma, y_uma = load_uma_test_data()
    if X_uma is None:
        return
    
    # Load feature selections
    try:
        with open('selected_features.json', 'r') as f:
            feature_selections = json.load(f)
    except:
        print("❌ Could not load feature selections")
        return
    
    # Load baseline UMA results for comparison
    try:
        with open('uma_test_results.json', 'r') as f:
            baseline_results = json.load(f)
        baseline_accuracy = baseline_results['accuracy']
    except:
        baseline_accuracy = 0.3049  # Known baseline
    
    print(f"📊 Baseline UMA Accuracy: {baseline_accuracy:.4f}")
    
    # Test each model
    models_to_test = [
        ('Domain-Robust', 'rf_domain_adapted.pkl', 'domain_robust'),
        ('Standard', 'rf_standard_baseline.pkl', 'standard'), 
        ('Conservative', 'rf_conservative.pkl', 'conservative')
    ]
    
    results = {}
    
    for model_name, model_file, feature_key in models_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING {model_name.upper()} MODEL")
        print(f"{'='*60}")
        
        try:
            # Load model
            model = joblib.load(model_file)
            print(f"✓ Loaded {model_name} model")
            
            # Get feature selection
            selected_features = feature_selections[feature_key]
            print(f"✓ Using {len(selected_features)} selected features")
            
            # Filter UMA data to selected features
            available_features = [f for f in selected_features if f in X_uma.columns]
            missing_features = [f for f in selected_features if f not in X_uma.columns]
            
            print(f"📊 Feature availability: {len(available_features)}/{len(selected_features)} features available")
            
            if len(missing_features) > 0:
                print(f"⚠️ Missing {len(missing_features)} features - using available ones")
            
            # Create test data
            X_uma_test = X_uma[available_features]
            
            # Make predictions
            print("🔮 Making predictions...")
            y_pred = model.predict(X_uma_test)
            accuracy = accuracy_score(y_uma, y_pred)
            
            print(f"✓ {model_name} UMA Accuracy: {accuracy:.4f}")
            
            # Improvement over baseline
            improvement = accuracy - baseline_accuracy
            percent_improvement = (improvement / baseline_accuracy) * 100
            
            print(f"📈 Improvement over baseline: +{improvement:.4f} ({percent_improvement:+.1f}%)")
            
            # Detailed classification report
            print(f"\n📊 {model_name} Classification Report:")
            print(classification_report(y_uma, y_pred))
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'improvement': improvement,
                'percent_improvement': percent_improvement,
                'available_features': len(available_features),
                'total_features': len(selected_features),
                'classification_report': classification_report(y_uma, y_pred, output_dict=True)
            }
            
        except Exception as e:
            print(f"❌ Error testing {model_name} model: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comparison visualization
    if results:
        print(f"\n{'='*60}")
        print("RESULTS COMPARISON")
        print(f"{'='*60}")
        
        # Results table
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print("-" * 70)
        print(f"{'Model':<15} {'UMA Acc':<10} {'Improvement':<12} {'Features':<10}")
        print("-" * 70)
        print(f"{'Baseline':<15} {baseline_accuracy:<10.4f} {'--':<12} {'4071':<10}")
        
        best_model = None
        best_accuracy = baseline_accuracy
        
        for model_name, metrics in results.items():
            accuracy = metrics['accuracy']
            improvement = metrics['improvement']
            features = f"{metrics['available_features']}/{metrics['total_features']}"
            
            print(f"{model_name:<15} {accuracy:<10.4f} {improvement:+8.4f} ({metrics['percent_improvement']:+5.1f}%) {features:<10}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        print("-" * 70)
        
        if best_model:
            print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        # Confusion matrices comparison
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
        if len(results) == 1:
            axes = [axes]
        
        for i, (model_name, metrics) in enumerate(results.items()):
            # Recreate predictions for confusion matrix
            try:
                model_file = [f for n, f, k in models_to_test if n == model_name][0]
                feature_key = [k for n, f, k in models_to_test if n == model_name][0]
                
                model = joblib.load(model_file)
                selected_features = feature_selections[feature_key]
                available_features = [f for f in selected_features if f in X_uma.columns]
                X_uma_test = X_uma[available_features]
                y_pred = model.predict(X_uma_test)
                
                cm = confusion_matrix(y_uma, y_pred)
                labels = sorted(y_uma.unique())
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=labels, yticklabels=labels, ax=axes[i])
                axes[i].set_title(f'{model_name}\nAcc: {metrics["accuracy"]:.3f}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('True')
                
            except Exception as e:
                print(f"Error creating confusion matrix for {model_name}: {e}")
        
        plt.tight_layout()
        plt.savefig('domain_adaptation_uma_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results
        final_results = {
            'experiment_info': {
                'baseline_accuracy': baseline_accuracy,
                'uma_test_size': len(y_uma),
                'uma_class_distribution': y_uma.value_counts().to_dict(),
                'best_model': best_model,
                'best_accuracy': best_accuracy
            },
            'model_results': results
        }
        
        with open('domain_adaptation_uma_test_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Results saved to 'domain_adaptation_uma_test_results.json'")
        
        # Summary and insights
        print(f"\n{'='*60}")
        print("SUMMARY AND INSIGHTS")
        print(f"{'='*60}")
        
        total_improvement = best_accuracy - baseline_accuracy if best_model else 0
        
        print(f"\n📋 EXPERIMENT SUMMARY:")
        print(f"  • Original Baseline: {baseline_accuracy:.4f}")
        if best_model:
            print(f"  • Best Domain-Adapted: {best_accuracy:.4f} ({best_model})")
            print(f"  • Total Improvement: +{total_improvement:.4f} ({(total_improvement/baseline_accuracy)*100:+.1f}%)")
        
        print(f"\n💡 KEY INSIGHTS:")
        
        if total_improvement > 0.1:
            print("  ✅ EXCELLENT improvement achieved!")
            print("  → Domain adaptation successfully reduced dataset bias")
            print("  → Model is much more robust for cross-dataset deployment")
        elif total_improvement > 0.05:
            print("  ✅ GOOD improvement achieved")
            print("  → Domain adaptation is working effectively")
            print("  → Consider ensemble approaches for further gains")
        elif total_improvement > 0.01:
            print("  ⚠️  MODEST improvement achieved")
            print("  → Some benefit from domain adaptation")
            print("  → May need more advanced techniques")
        else:
            print("  ❌ LIMITED improvement achieved")
            print("  → Dataset bias remains strong")
            print("  → Consider joint training or data augmentation")
        
        if best_model:
            best_metrics = results[best_model]
            feature_ratio = best_metrics['available_features'] / best_metrics['total_features']
            
            print(f"\n🎯 BEST MODEL ANALYSIS ({best_model}):")
            print(f"  • Feature utilization: {feature_ratio:.1%}")
            print(f"  • Performance gain: {best_metrics['percent_improvement']:+.1f}%")
            
            # Class-wise performance
            cr = best_metrics['classification_report']
            print(f"  • Class-wise F1 scores:")
            for class_name in ['Fall', 'Walking', 'Jogging', 'Stairs']:
                if class_name in cr:
                    f1 = cr[class_name]['f1-score']
                    print(f"    - {class_name}: {f1:.3f}")
        
        print(f"\n🔄 NEXT STEPS:")
        print("  1. Deploy best model for cross-dataset scenarios")
        print("  2. Consider ensemble of domain-adapted models")
        print("  3. Collect more diverse training data")
        print("  4. Explore advanced domain adaptation techniques")
        
        return final_results
    
    else:
        print("❌ No models were successfully tested")
        return None

if __name__ == "__main__":
    results = test_domain_adapted_models()
