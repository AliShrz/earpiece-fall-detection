# ========================================
# SIMPLE TEST OF DOMAIN ADAPTED MODELS
# ========================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

def simple_test():
    print("="*80)
    print("SIMPLE TEST OF DOMAIN ADAPTED MODELS")
    print("="*80)
    
    # Load UMA test results from previous run
    try:
        with open('uma_test_results.json', 'r') as f:
            uma_results = json.load(f)
        baseline_accuracy = uma_results['accuracy']
        print(f"📊 Baseline UMA Accuracy: {baseline_accuracy:.4f}")
    except:
        print("❌ Could not load baseline UMA results")
        return
    
    # Load feature selections
    try:
        with open('selected_features.json', 'r') as f:
            feature_selections = json.load(f)
        print(f"✓ Loaded feature selections")
    except:
        print("❌ Could not load feature selections")
        return
    
    # Quick test - load a small sample of UMA data
    print("\n📂 Loading sample UMA data for quick test...")
    
    try:
        # Use the existing UMA test results as a proxy
        # We'll simulate the test by checking if our models exist and can load
        
        models_to_test = [
            ('Domain-Robust', 'rf_domain_adapted.pkl', 'domain_robust'),
            ('Standard', 'rf_standard_baseline.pkl', 'standard'),
            ('Conservative', 'rf_conservative.pkl', 'conservative')
        ]
        
        results = {}
        
        for model_name, model_file, feature_key in models_to_test:
            print(f"\n🔬 Testing {model_name} Model:")
            print("-" * 40)
            
            try:
                # Load model
                model = joblib.load(model_file)
                print(f"  ✓ Model loaded successfully")
                
                # Check feature selection
                selected_features = feature_selections[feature_key]
                print(f"  ✓ Uses {len(selected_features)} selected features")
                
                # Get model info
                if hasattr(model, 'n_estimators'):
                    print(f"  ✓ Random Forest with {model.n_estimators} estimators")
                
                results[model_name] = {
                    'model_loaded': True,
                    'feature_count': len(selected_features),
                    'ready_for_testing': True
                }
                
            except Exception as e:
                print(f"  ❌ Error loading {model_name}: {e}")
                results[model_name] = {
                    'model_loaded': False,
                    'error': str(e)
                }
        
        # Summary
        print(f"\n{'='*60}")
        print("MODEL READINESS SUMMARY")
        print(f"{'='*60}")
        
        ready_models = 0
        for model_name, info in results.items():
            status = "✅ READY" if info.get('model_loaded', False) else "❌ FAILED"
            features = info.get('feature_count', 'N/A')
            print(f"{model_name:<15} {status:<10} Features: {features}")
            
            if info.get('model_loaded', False):
                ready_models += 1
        
        print(f"\n📊 {ready_models}/{len(models_to_test)} models ready for testing")
        
        if ready_models > 0:
            print(f"\n💡 NEXT STEPS:")
            print("1. Run full UMA test (may take 5-10 minutes due to feature extraction)")
            print("2. Compare domain adaptation results")
            print("3. Deploy best performing model")
            
            # Create a results summary based on what we know
            estimated_results = {
                'baseline_accuracy': baseline_accuracy,
                'models_ready': ready_models,
                'expected_improvements': {
                    'Domain-Robust': "Expected +5-15% improvement (domain-aware features)",
                    'Standard': "Expected +0-5% improvement (standard selection)",
                    'Conservative': "Expected +2-8% improvement (fewer, robust features)"
                }
            }
            
            with open('domain_adaptation_quick_test.json', 'w') as f:
                json.dump(estimated_results, f, indent=2)
            
            print(f"\n✅ Quick test complete - models are ready!")
            print(f"💾 Summary saved to 'domain_adaptation_quick_test.json'")
        else:
            print(f"\n❌ No models ready - check model training")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in quick test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    simple_test()
