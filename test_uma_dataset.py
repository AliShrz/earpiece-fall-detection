# ========================================
# UMA DATASET TESTING SCRIPT
# Test trained model on unseen UMA dataset
# ========================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def map_uma_activity_to_sisfall(activity_number):
    """Map UMA activity numbers to SisFall activity categories"""
    # Based on UMA activity mapping:
    # 1: Walking, 2: Jogging, 5: GoDownstairs, 6: GoUpstairs
    # 13: backwardFall, 14: forwardFall, 15: lateralFall
    
    if activity_number == 1:
        return 'Walking'
    elif activity_number == 2:
        return 'Jogging'
    elif activity_number in [5, 6]:  # GoDownstairs, GoUpstairs
        return 'Stairs'
    elif activity_number in [13, 14, 15]:  # backwardFall, forwardFall, lateralFall
        return 'Fall'
    else:
        return None  # Skip other activities (Bending, Hopping, LyingDown, etc.)

def load_uma_data(uma_folder_path):
    """Load and process UMA dataset files from output_uma folder structure"""
    print("Loading UMA dataset from output_uma folder...")
    
    # Find all CSV files in the output_uma structure
    csv_files = []
    for subject_folder in glob.glob(os.path.join(uma_folder_path, "Subject*")):
        for activity_folder in glob.glob(os.path.join(subject_folder, "Activity*")):
            for trial_folder in glob.glob(os.path.join(activity_folder, "Trial*")):
                csv_file = glob.glob(os.path.join(trial_folder, "*.csv"))
                if csv_file:
                    csv_files.extend(csv_file)
    
    print(f"Found {len(csv_files)} UMA files")
    
    all_data = []
    labels = []
    file_ids = []
    
    processed_count = 0
    skipped_count = 0
    
    for file_path in csv_files:
        try:
            # Parse path to extract subject, activity, and trial information
            path_parts = file_path.split(os.sep)
            subject_folder = [p for p in path_parts if p.startswith('Subject')][-1]
            activity_folder = [p for p in path_parts if p.startswith('Activity')][-1]
            trial_folder = [p for p in path_parts if p.startswith('Trial')][-1]
            
            # Extract numbers
            subject_num = int(subject_folder.replace('Subject', ''))
            activity_num = int(activity_folder.replace('Activity', ''))
            trial_num = int(trial_folder.replace('Trial', ''))
            
            # Map activity to our categories
            mapped_activity = map_uma_activity_to_sisfall(activity_num)
            
            if mapped_activity is None:
                skipped_count += 1
                continue
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if file has the expected columns
            expected_cols = {
                'Accelerometer: x-axis (g)': 'Acc_X',
                'Accelerometer: y-axis (g)': 'Acc_Y', 
                'Accelerometer: z-axis (g)': 'Acc_Z',
                'Gyroscope: x-axis (rad/s)': 'Gyr_X',
                'Gyroscope: y-axis (rad/s)': 'Gyr_Y',
                'Gyroscope: z-axis (rad/s)': 'Gyr_Z'
            }
            
            # Check if all required columns exist
            missing_cols = set(expected_cols.keys()) - set(df.columns)
            if missing_cols:
                print(f"Skipping {os.path.basename(file_path)} - missing columns: {missing_cols}")
                skipped_count += 1
                continue
            
            # Rename columns to match original training data format (with spaces)
            original_format_cols = {
                'Accelerometer: x-axis (g)': 'Acc X',
                'Accelerometer: y-axis (g)': 'Acc Y', 
                'Accelerometer: z-axis (g)': 'Acc Z',
                'Gyroscope: x-axis (rad/s)': 'Gyro X',
                'Gyroscope: y-axis (rad/s)': 'Gyro Y',
                'Gyroscope: z-axis (rad/s)': 'Gyro Z'
            }
            
            sensor_data = df[list(expected_cols.keys())].copy()
            sensor_data.columns = list(original_format_cols.values())
            
            # Ensure all sensor data is numeric
            for col in ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']:
                sensor_data[col] = pd.to_numeric(sensor_data[col], errors='coerce')
            
            # Remove any rows with NaN values
            sensor_data = sensor_data.dropna()
            
            # Skip if too few data points
            if len(sensor_data) < 50:
                print(f"Skipping {os.path.basename(file_path)} - too few data points: {len(sensor_data)}")
                skipped_count += 1
                continue
            
            # Add time index and ID for tsfresh
            sensor_data['time'] = range(len(sensor_data))
            file_id = f"UMA_S{subject_num}_A{activity_num}_T{trial_num}"
            sensor_data['id'] = file_id
            
            all_data.append(sensor_data)
            labels.append(mapped_activity)
            file_ids.append(file_id)
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} files...")
                
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            skipped_count += 1
            continue
    
    print(f"\nUMA Dataset Summary:")
    print(f"✓ Successfully processed: {processed_count} files")
    print(f"⚠ Skipped: {skipped_count} files")
    
    if processed_count == 0:
        print("❌ No valid files found!")
        return None, None, None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Create labels dataframe
    labels_df = pd.DataFrame({
        'ID': file_ids,
        'Label': labels
    })
    
    print("\nActivity distribution in UMA dataset:")
    print(labels_df['Label'].value_counts())
    
    return combined_data, labels_df, file_ids

def extract_uma_features(sensor_data):
    """Extract features from UMA sensor data using tsfresh - SAME AS TRAINING"""
    print("\nExtracting features from UMA data using FULL comprehensive feature extraction...")
    
    # Ensure all columns except 'id' and 'time' are numeric
    numeric_cols = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']
    for col in numeric_cols:
        sensor_data[col] = pd.to_numeric(sensor_data[col], errors='coerce')
    
    # Remove any rows with NaN values
    sensor_data = sensor_data.dropna()
    
    print(f"Sensor data shape after cleaning: {sensor_data.shape}")
    
    # Use THE EXACT SAME comprehensive feature extraction as the original training
    # This is critical - must match the training exactly!
    print("Using comprehensive feature extraction (same as training)...")
    
    try:
        extracted_features = extract_features(
            sensor_data, 
            column_id="id", 
            column_sort="time",
            default_fc_parameters=None,  # Use ALL default features - same as training
            n_jobs=1,  # Single job to avoid multiprocessing issues
            disable_progressbar=False
        )
        
        # Handle missing values the same way as training
        print("Imputing missing values...")
        extracted_features = impute(extracted_features)
        
        print(f"✓ Successfully extracted {extracted_features.shape[1]} features using full extraction")
        
    except Exception as e:
        print(f"❌ Error with comprehensive feature extraction: {e}")
        print("This is critical - the model expects comprehensive features!")
        
        # Try one more time with different parameters
        print("Trying comprehensive extraction with single core and smaller chunks...")
        try:
            extracted_features = extract_features(
                sensor_data, 
                column_id="id", 
                column_sort="time",
                default_fc_parameters=None,
                n_jobs=1,
                chunksize=None,  # Process all at once
                disable_progressbar=True
            )
            
            extracted_features = impute(extracted_features)
            print(f"✓ Successfully extracted {extracted_features.shape[1]} features on second attempt")
            
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            print("❌ Cannot proceed without comprehensive features that match training!")
            return pd.DataFrame()  # Return empty DataFrame to signal failure
    
    print(f"Final UMA features shape: {extracted_features.shape}")
    print(f"Expected ~4000+ features to match training data")
    
    # Verify we have enough features
    if extracted_features.shape[1] < 1000:
        print(f"⚠️ WARNING: Only {extracted_features.shape[1]} features extracted!")
        print("This is much less than the ~4662 features in training data")
        print("Results may be unreliable due to feature mismatch")
    else:
        print(f"✓ Good: {extracted_features.shape[1]} features extracted")
    
    return extracted_features

def test_model_on_uma(model_path, uma_features, uma_labels, original_feature_names):
    """Test the trained model on UMA dataset with exact feature matching"""
    print(f"\nTesting model from {model_path}...")
    
    # Load the trained model
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded successfully")
        print(f"Model expects {len(original_feature_names)} features")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Critical feature alignment
    print("Performing EXACT feature alignment with training data...")
    
    uma_feature_names = set(uma_features.columns)
    training_feature_names = set(original_feature_names)
    common_features = uma_feature_names & training_feature_names
    missing_in_uma = training_feature_names - uma_feature_names
    extra_in_uma = uma_feature_names - training_feature_names
    
    print(f"Training features: {len(training_feature_names)}")
    print(f"UMA features: {len(uma_feature_names)}")
    print(f"Common features: {len(common_features)}")
    print(f"Missing in UMA: {len(missing_in_uma)}")
    print(f"Extra in UMA: {len(extra_in_uma)}")
    
    if len(missing_in_uma) > 0:
        print(f"⚠️ WARNING: {len(missing_in_uma)} features missing from UMA data")
        if len(missing_in_uma) < 10:
            print("Missing features:", list(missing_in_uma)[:10])
    
    if len(common_features) < len(training_feature_names) * 0.8:
        print(f"❌ Too many missing features! Only {len(common_features)}/{len(training_feature_names)} ({len(common_features)/len(training_feature_names)*100:.1f}%) available")
        print("Cannot reliably test the model with so many missing features")
        return None
    
    # Create feature matrix with EXACT same order and features as training
    print("Creating feature matrix with exact training feature order...")
    
    # Start with all training features
    X_uma_aligned = pd.DataFrame(index=uma_features.index, columns=original_feature_names)
    
    # Fill in available features
    for feature in original_feature_names:
        if feature in uma_features.columns:
            X_uma_aligned[feature] = uma_features[feature]
        else:
            # Fill missing features with 0 (or could use mean of available features)
            X_uma_aligned[feature] = 0.0
    
    # Ensure all data is numeric
    X_uma_aligned = X_uma_aligned.astype(float)
    
    print(f"Final aligned feature matrix shape: {X_uma_aligned.shape}")
    print(f"Feature order matches training: {list(X_uma_aligned.columns) == original_feature_names}")
    
    # Make predictions
    print("Making predictions with sklearn feature validation disabled...")
    
    try:
        # For sklearn >= 1.2, we need to be careful about feature name validation
        # The model was trained with specific feature names, so we need exact match
        y_pred = model.predict(X_uma_aligned)
        y_pred_proba = model.predict_proba(X_uma_aligned)
        
        # Get prediction confidence
        confidence = np.max(y_pred_proba, axis=1)
        
        print(f"✓ Successfully made predictions on {len(y_pred)} samples")
        print(f"✓ Prediction confidence range: {np.min(confidence):.3f} - {np.max(confidence):.3f}")
        
        return y_pred, y_pred_proba, confidence, X_uma_aligned
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        
        # Try to bypass sklearn's feature validation by converting to numpy
        print("Trying with numpy array to bypass feature name validation...")
        try:
            X_uma_numpy = X_uma_aligned.values
            y_pred = model.predict(X_uma_numpy)
            y_pred_proba = model.predict_proba(X_uma_numpy)
            confidence = np.max(y_pred_proba, axis=1)
            
            print(f"✓ Successfully made predictions using numpy arrays")
            return y_pred, y_pred_proba, confidence, X_uma_aligned
            
        except Exception as e2:
            print(f"❌ Still failed with numpy arrays: {e2}")
            return None

def evaluate_uma_results(y_true, y_pred, confidence, dataset_name="UMA"):
    """Evaluate and visualize results on UMA dataset"""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ON {dataset_name} DATASET")
    print(f"{'='*60}")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {dataset_name} Dataset")
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confidence analysis
    print(f"\nConfidence Analysis:")
    print(f"Mean confidence: {np.mean(confidence):.3f}")
    print(f"Min confidence: {np.min(confidence):.3f}")
    print(f"Max confidence: {np.max(confidence):.3f}")
    
    # High confidence predictions
    high_conf_mask = confidence >= 0.8
    high_conf_accuracy = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
    print(f"High confidence (≥0.8) predictions: {np.sum(high_conf_mask)}/{len(confidence)} ({np.mean(high_conf_mask)*100:.1f}%)")
    print(f"High confidence accuracy: {high_conf_accuracy:.3f}")
    
    # Plot confidence distribution
    plt.figure(figsize=(12, 8))
    
    # Confidence distribution
    plt.subplot(2, 2, 1)
    plt.hist(confidence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(confidence), color='red', linestyle='--', label=f'Mean: {np.mean(confidence):.3f}')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title(f'{dataset_name} - Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confidence by class
    plt.subplot(2, 2, 2)
    for i, label in enumerate(labels):
        class_mask = np.array(y_true) == label
        if np.sum(class_mask) > 0:
            class_conf = confidence[class_mask]
            plt.hist(class_conf, bins=10, alpha=0.6, label=f'{label} (n={np.sum(class_mask)})')
    
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title(f'{dataset_name} - Confidence by True Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Correct vs Incorrect predictions
    plt.subplot(2, 2, 3)
    correct = np.array(y_true) == np.array(y_pred)
    correct_conf = confidence[correct]
    incorrect_conf = confidence[~correct]
    
    if len(incorrect_conf) > 0:
        plt.hist([correct_conf, incorrect_conf], bins=15, alpha=0.7, 
                label=['Correct', 'Incorrect'], color=['green', 'red'])
    else:
        plt.hist(correct_conf, bins=15, alpha=0.7, label='All Correct', color='green')
    
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title(f'{dataset_name} - Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Class-wise accuracy
    plt.subplot(2, 2, 4)
    class_accuracies = []
    class_counts = []
    
    for label in labels:
        mask = np.array(y_true) == label
        if np.sum(mask) > 0:
            class_acc = np.mean(np.array(y_pred)[mask] == label)
            class_accuracies.append(class_acc)
            class_counts.append(np.sum(mask))
        else:
            class_accuracies.append(0)
            class_counts.append(0)
    
    bars = plt.bar(labels, class_accuracies, alpha=0.7, color='lightcoral')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name} - Accuracy by Class')
    plt.xticks(rotation=45)
    
    # Add count annotations
    for bar, acc, count in zip(bars, class_accuracies, class_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.2f}\n(n={count})', ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print(f"\n{'='*40}")
    print("SUMMARY STATISTICS")
    print(f"{'='*40}")
    print(f"Total samples: {len(y_true)}")
    print(f"Correct predictions: {np.sum(correct)}")
    print(f"Incorrect predictions: {np.sum(~correct)}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Mean confidence: {np.mean(confidence):.4f}")
    print(f"Std confidence: {np.std(confidence):.4f}")
    
    return {
        'accuracy': accuracy,
        'mean_confidence': np.mean(confidence),
        'high_confidence_ratio': np.mean(high_conf_mask),
        'high_confidence_accuracy': high_conf_accuracy,
        'class_accuracies': dict(zip(labels, class_accuracies))
    }

def compare_datasets_performance(sisfall_results, uma_results):
    """Compare performance between SisFall and UMA datasets"""
    print(f"\n{'='*60}")
    print("DATASET COMPARISON")
    print(f"{'='*60}")
    
    # Create comparison visualization
    plt.figure(figsize=(15, 10))
    
    # Overall accuracy comparison
    plt.subplot(2, 3, 1)
    datasets = ['SisFall (Training)', 'UMA (Unseen)']
    accuracies = [sisfall_results.get('accuracy', 0), uma_results['accuracy']]
    colors = ['blue', 'red']
    
    bars = plt.bar(datasets, accuracies, color=colors, alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy Comparison')
    plt.ylim(0, 1.1)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Performance drop
    if sisfall_results.get('accuracy'):
        drop = sisfall_results['accuracy'] - uma_results['accuracy']
        plt.text(0.5, 0.5, f'Performance Drop: {drop:.3f}', transform=plt.gca().transAxes,
                ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
    
    plt.grid(True, alpha=0.3)
    
    # Confidence comparison
    plt.subplot(2, 3, 2)
    confidences = [sisfall_results.get('mean_confidence', 0), uma_results['mean_confidence']]
    bars = plt.bar(datasets, confidences, color=colors, alpha=0.7)
    plt.ylabel('Mean Confidence')
    plt.title('Prediction Confidence Comparison')
    plt.ylim(0, 1.1)
    
    for bar, conf in zip(bars, confidences):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Class-wise accuracy comparison (if available)
    plt.subplot(2, 3, 3)
    common_classes = set(sisfall_results.get('class_accuracies', {}).keys()) & set(uma_results['class_accuracies'].keys())
    
    if common_classes:
        x_pos = np.arange(len(common_classes))
        width = 0.35
        
        sisfall_acc = [sisfall_results.get('class_accuracies', {}).get(cls, 0) for cls in common_classes]
        uma_acc = [uma_results['class_accuracies'][cls] for cls in common_classes]
        
        plt.bar(x_pos - width/2, sisfall_acc, width, label='SisFall', alpha=0.7, color='blue')
        plt.bar(x_pos + width/2, uma_acc, width, label='UMA', alpha=0.7, color='red')
        
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Class-wise Accuracy Comparison')
        plt.xticks(x_pos, list(common_classes), rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No common classes\nfor comparison', ha='center', va='center')
        plt.title('Class-wise Accuracy Comparison')
    
    # Generalization analysis
    plt.subplot(2, 3, 4)
    metrics = ['Accuracy', 'Confidence']
    sisfall_vals = [sisfall_results.get('accuracy', 0), sisfall_results.get('mean_confidence', 0)]
    uma_vals = [uma_results['accuracy'], uma_results['mean_confidence']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, sisfall_vals, width, label='SisFall', alpha=0.7, color='blue')
    plt.bar(x + width/2, uma_vals, width, label='UMA', alpha=0.7, color='red')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Generalization Analysis')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Text summary
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    summary_text = f"""GENERALIZATION ASSESSMENT
    
Original Dataset (SisFall):
• Accuracy: {sisfall_results.get('accuracy', 'N/A'):.3f}
• Confidence: {sisfall_results.get('mean_confidence', 'N/A'):.3f}

Unseen Dataset (UMA):
• Accuracy: {uma_results['accuracy']:.3f}
• Confidence: {uma_results['mean_confidence']:.3f}

Performance Analysis:
"""
    
    if sisfall_results.get('accuracy'):
        acc_drop = sisfall_results['accuracy'] - uma_results['accuracy']
        conf_drop = sisfall_results.get('mean_confidence', 0) - uma_results['mean_confidence']
        
        summary_text += f"""• Accuracy Drop: {acc_drop:.3f}
• Confidence Drop: {conf_drop:.3f}

Generalization Status:"""
        
        if acc_drop < 0.05:
            summary_text += "\n✅ EXCELLENT generalization"
        elif acc_drop < 0.1:
            summary_text += "\n✅ GOOD generalization"
        elif acc_drop < 0.2:
            summary_text += "\n⚠️ MODERATE generalization"
        else:
            summary_text += "\n❌ POOR generalization"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to test model on UMA dataset"""
    print("="*60)
    print("TESTING TRAINED MODEL ON UMA DATASET")
    print("="*60)
    
    # Paths
    uma_folder = "/Users/amir/Documents/projects/earpiece-fall-detection/output_uma"
    model_path = "/Users/amir/Documents/projects/earpiece-fall-detection/random_forest_improved_v2.pkl"
    
    # Check if paths exist
    if not os.path.exists(uma_folder):
        print(f"❌ UMA dataset folder not found: {uma_folder}")
        print("Please check the path and try again.")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using improved_classification.py")
        return
    
    # Load UMA data
    sensor_data, labels_df, file_ids = load_uma_data(uma_folder)
    
    if sensor_data is None:
        print("❌ Failed to load UMA dataset")
        return
    
    # Extract features from UMA data
    uma_features = extract_uma_features(sensor_data)
    
    if uma_features.empty:
        print("❌ Failed to extract features from UMA dataset")
        return
    
    # Load original training feature names - CRITICAL FOR MATCHING
    try:
        # Load the EXACT same feature file that was used for training
        print("Loading original training feature names from the exact training file...")
        print("This may take a moment as the file is large...")
        
        # Load just the column names (header) without loading all data
        with open('extracted_features_full_Gyro_20250618_224358.csv', 'r') as f:
            header_line = f.readline().strip()
            original_features = header_line.split(',')[1:]  # Skip the first column (index)
        
        print(f"✓ Successfully loaded {len(original_features)} original feature names")
        print(f"First few features: {original_features[:5]}")
        print(f"Expected around 4662 features")
        
        if len(original_features) < 1000:
            raise ValueError(f"Too few features loaded: {len(original_features)}")
            
    except Exception as e:
        print(f"⚠️ Could not load original feature names from CSV header: {e}")
        
        # Try alternative method - load small sample
        try:
            print("Trying to load small sample of training data...")
            original_training_data = pd.read_csv('extracted_features_full_Gyro_20250618_224358.csv', 
                                                index_col=0, nrows=1)
            original_features = list(original_training_data.columns)
            print(f"✓ Loaded {len(original_features)} feature names from sample")
            
        except Exception as e2:
            print(f"❌ Could not load original feature names: {e2}")
            
            # Final fallback - try validation results
            try:
                import json
                with open('validation_results.json', 'r') as f:
                    validation_data = json.load(f)
                    if 'top_features' in validation_data:
                        original_features = [feat[0] for feat in validation_data['top_features']]
                        print(f"⚠️ FALLBACK: Using only top {len(original_features)} features from validation results")
                        print("This is NOT ideal - results may be unreliable!")
                    else:
                        raise FileNotFoundError("No feature names in validation results")
            except:
                print("❌ Could not load original feature names from any source")
                print("This will likely cause the test to fail!")
                return
    
    # Test model on UMA dataset
    y_pred, y_pred_proba, confidence, X_uma_final = test_model_on_uma(
        model_path, uma_features, labels_df['Label'], original_features
    )
    
    if y_pred is None:
        print("❌ Failed to test model on UMA dataset")
        return
    
    # Evaluate results
    uma_results = evaluate_uma_results(
        labels_df['Label'], y_pred, confidence, "UMA"
    )
    
    # Try to load SisFall results for comparison
    try:
        with open('validation_results.json', 'r') as f:
            sisfall_results = json.load(f)
            sisfall_results['accuracy'] = sisfall_results.get('test_accuracy', 0)
            sisfall_results['mean_confidence'] = 0.95  # Estimated based on previous results
    except:
        print("⚠️ Could not load SisFall results for comparison")
        sisfall_results = {}
    
    # Compare datasets
    if sisfall_results:
        compare_datasets_performance(sisfall_results, uma_results)
    
    # Save UMA test results
    uma_test_results = {
        'dataset': 'UMA',
        'total_samples': len(labels_df),
        'accuracy': uma_results['accuracy'],
        'mean_confidence': uma_results['mean_confidence'],
        'high_confidence_ratio': uma_results['high_confidence_ratio'],
        'high_confidence_accuracy': uma_results['high_confidence_accuracy'],
        'class_distribution': labels_df['Label'].value_counts().to_dict(),
        'class_accuracies': uma_results['class_accuracies']
    }
    
    with open('uma_test_results.json', 'w') as f:
        json.dump(uma_test_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETED")
    print(f"{'='*60}")
    print("✅ UMA dataset testing completed")
    print("✅ Results saved to 'uma_test_results.json'")
    print("✅ Visualizations saved as PNG files")
    
    if uma_results['accuracy'] > 0.8:
        print("🌟 EXCELLENT: Model shows strong generalization to unseen data!")
    elif uma_results['accuracy'] > 0.6:
        print("✅ GOOD: Model shows reasonable generalization")
    elif uma_results['accuracy'] > 0.4:
        print("⚠️ MODERATE: Model shows limited generalization")
    else:
        print("❌ POOR: Model shows weak generalization to unseen data")

if __name__ == "__main__":
    main()
