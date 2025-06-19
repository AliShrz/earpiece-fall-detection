import numpy as np
import pandas as pd
import os
import glob
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class CustomFeatureExtractor:
    """Extract custom features from accelerometer and gyroscope data"""
    
    def __init__(self, sampling_rate=200):
        self.sampling_rate = sampling_rate
    
    def mean(self, signal):
        """Mean value"""
        return np.mean(signal)
    
    def std(self, signal):
        """Standard deviation"""
        return np.std(signal)
    
    def mad(self, signal):
        """Median absolute deviation"""
        return np.median(np.abs(signal - np.median(signal)))
    
    def max_val(self, signal):
        """Maximum value"""
        return np.max(signal)
    
    def min_val(self, signal):
        """Minimum value"""
        return np.min(signal)
    
    def sma(self, acc_x, acc_y, acc_z):
        """Signal Magnitude Area"""
        return np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z))
    
    def energy(self, signal):
        """Energy measure - sum of squares divided by number of values"""
        return np.sum(signal**2) / len(signal)
    
    def iqr(self, signal):
        """Interquartile range"""
        q75, q25 = np.percentile(signal, [75, 25])
        return q75 - q25
    
    def entropy(self, signal):
        """Signal entropy"""
        # Normalize signal to probabilities
        signal_abs = np.abs(signal)
        if np.sum(signal_abs) == 0:
            return 0
        probs = signal_abs / np.sum(signal_abs)
        probs = probs[probs > 0]  # Remove zeros to avoid log(0)
        return -np.sum(probs * np.log2(probs))
    
    def ar_coeff(self, signal, order=4):
        """Autoregression coefficients using Yule-Walker method"""
        try:
            # Remove mean
            signal_centered = signal - np.mean(signal)
            # Calculate autocorrelation
            autocorr = np.correlate(signal_centered, signal_centered, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Solve Yule-Walker equations
            if len(autocorr) > order:
                R = np.array([autocorr[i:i+order] for i in range(order)])
                r = autocorr[1:order+1]
                coeffs = np.linalg.solve(R, r)
                return coeffs
            else:
                return np.zeros(order)
        except:
            return np.zeros(order)
    
    def correlation(self, signal1, signal2):
        """Correlation coefficient between two signals"""
        return np.corrcoef(signal1, signal2)[0, 1] if len(signal1) == len(signal2) else 0
    
    def max_inds(self, signal):
        """Index of frequency component with largest magnitude (normalized)"""
        fft_vals = np.abs(fft(signal))
        max_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1  # Exclude DC component
        return max_idx / (len(fft_vals)//2)  # Normalize by Nyquist
    
    def mean_freq(self, signal):
        """Weighted average of frequency components"""
        fft_vals = np.abs(fft(signal))
        freqs = fftfreq(len(signal), 1/self.sampling_rate)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_vals[:len(fft_vals)//2]
        
        if np.sum(positive_fft) == 0:
            return 0
        return np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
    
    def skewness(self, signal):
        """Skewness of the signal"""
        return stats.skew(signal)
    
    def kurtosis(self, signal):
        """Kurtosis of the signal"""
        return stats.kurtosis(signal)
    
    def bands_energy(self, signal, n_bands=8):
        """Energy of frequency intervals within FFT bins"""
        fft_vals = np.abs(fft(signal))**2
        positive_fft = fft_vals[:len(fft_vals)//2]
        
        band_size = len(positive_fft) // n_bands
        bands = []
        
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < n_bands - 1 else len(positive_fft)
            band_energy = np.sum(positive_fft[start_idx:end_idx])
            bands.append(band_energy)
        
        return np.array(bands)
    
    def angle(self, vec1, vec2):
        """Angle between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norms == 0:
            return 0
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def extract_all_features(self, data):
        """Extract all features from a data window"""
        features = {}
        
        # Extract individual sensor data
        acc_x_adxl = data[:, 0]  # ADXL345 accelerometer X
        acc_y_adxl = data[:, 1]  # ADXL345 accelerometer Y  
        acc_z_adxl = data[:, 2]  # ADXL345 accelerometer Z
        
        gyro_x = data[:, 3]      # ITG3200 gyroscope X
        gyro_y = data[:, 4]      # ITG3200 gyroscope Y
        gyro_z = data[:, 5]      # ITG3200 gyroscope Z
        
        acc_x_mma = data[:, 6]   # MMA8451Q accelerometer X
        acc_y_mma = data[:, 7]   # MMA8451Q accelerometer Y
        acc_z_mma = data[:, 8]   # MMA8451Q accelerometer Z
        
        # Calculate magnitude signals
        acc_mag_adxl = np.sqrt(acc_x_adxl**2 + acc_y_adxl**2 + acc_z_adxl**2)
        acc_mag_mma = np.sqrt(acc_x_mma**2 + acc_y_mma**2 + acc_z_mma**2)
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        
        # Extract features for each signal
        signals = {
            'acc_x_adxl': acc_x_adxl,
            'acc_y_adxl': acc_y_adxl,
            'acc_z_adxl': acc_z_adxl,
            'acc_mag_adxl': acc_mag_adxl,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'gyro_mag': gyro_mag,
            'acc_x_mma': acc_x_mma,
            'acc_y_mma': acc_y_mma,
            'acc_z_mma': acc_z_mma,
            'acc_mag_mma': acc_mag_mma
        }
        
        # Basic statistical features for each signal
        for name, signal in signals.items():
            features[f'{name}_mean'] = self.mean(signal)
            features[f'{name}_std'] = self.std(signal)
            features[f'{name}_mad'] = self.mad(signal)
            features[f'{name}_max'] = self.max_val(signal)
            features[f'{name}_min'] = self.min_val(signal)
            features[f'{name}_energy'] = self.energy(signal)
            features[f'{name}_iqr'] = self.iqr(signal)
            features[f'{name}_entropy'] = self.entropy(signal)
            features[f'{name}_skewness'] = self.skewness(signal)
            features[f'{name}_kurtosis'] = self.kurtosis(signal)
            features[f'{name}_max_inds'] = self.max_inds(signal)
            features[f'{name}_mean_freq'] = self.mean_freq(signal)
            
            # Autoregression coefficients
            ar_coeffs = self.ar_coeff(signal)
            for i, coeff in enumerate(ar_coeffs):
                features[f'{name}_ar_coeff_{i+1}'] = coeff
            
            # Frequency bands energy
            bands = self.bands_energy(signal)
            for i, band_energy in enumerate(bands):
                features[f'{name}_band_energy_{i+1}'] = band_energy
        
        # Signal Magnitude Area for accelerometers
        features['sma_adxl'] = self.sma(acc_x_adxl, acc_y_adxl, acc_z_adxl)
        features['sma_mma'] = self.sma(acc_x_mma, acc_y_mma, acc_z_mma)
        
        # Correlation features
        features['corr_acc_x_adxl_mma'] = self.correlation(acc_x_adxl, acc_x_mma)
        features['corr_acc_y_adxl_mma'] = self.correlation(acc_y_adxl, acc_y_mma)
        features['corr_acc_z_adxl_mma'] = self.correlation(acc_z_adxl, acc_z_mma)
        features['corr_acc_mag_adxl_mma'] = self.correlation(acc_mag_adxl, acc_mag_mma)
        
        # Angle features
        features['angle_acc_adxl_mma'] = self.angle(
            [self.mean(acc_x_adxl), self.mean(acc_y_adxl), self.mean(acc_z_adxl)],
            [self.mean(acc_x_mma), self.mean(acc_y_mma), self.mean(acc_z_mma)]
        )
        
        return features

class SisFallDataLoader:
    """Load and process SisFall dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.activity_mapping = {
            # Walking activities
            'D01': 'walking',    # Walking slowly
            'D02': 'walking',    # Walking quickly
            
            # Jogging activities  
            'D03': 'jogging',    # Jogging slowly
            'D04': 'jogging',    # Jogging quickly
            
            # Sitting activities
            'D07': 'sitting',    # Slowly sit in half height chair
            'D08': 'sitting',    # Quickly sit in half height chair
            'D09': 'sitting',    # Slowly sit in low height chair
            'D10': 'sitting',    # Quickly sit in low height chair
            'D11': 'sitting',    # Sitting, trying to get up, collapse
            'D12': 'sitting',    # Sitting, lying slowly, sit again
            'D13': 'sitting',    # Sitting, lying quickly, sit again
            
            # Stairs activities
            'D05': 'stairs',     # Walking upstairs/downstairs slowly
            'D06': 'stairs',     # Walking upstairs/downstairs quickly
            
            # All fall activities
            'F01': 'falls', 'F02': 'falls', 'F03': 'falls', 'F04': 'falls', 'F05': 'falls',
            'F06': 'falls', 'F07': 'falls', 'F08': 'falls', 'F09': 'falls', 'F10': 'falls',
            'F11': 'falls', 'F12': 'falls', 'F13': 'falls', 'F14': 'falls', 'F15': 'falls'
        }
    
    def load_file(self, filepath):
        """Load a single data file"""
        try:
            # Read the file and parse the data properly
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            data_rows = []
            for line in lines:
                # Remove semicolon and split by comma
                line = line.strip().rstrip(';')
                if line:
                    values = [float(x.strip()) for x in line.split(',')]
                    if len(values) == 9:  # Ensure we have all 9 sensor values
                        data_rows.append(values)
            
            if len(data_rows) == 0:
                print(f"No valid data rows found in {filepath}")
                return None
                
            return np.array(data_rows)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def get_activity_label(self, filename):
        """Extract activity label from filename"""
        activity_code = filename.split('_')[0]
        return self.activity_mapping.get(activity_code, 'unknown')
    
    def load_subject_data(self, subject_folder, window_size=400, overlap=200):
        """Load all data for a subject with sliding window"""
        subject_path = os.path.join(self.dataset_path, subject_folder)
        if not os.path.exists(subject_path):
            print(f"Subject folder {subject_folder} not found")
            return [], []
        
        all_features = []
        all_labels = []
        
        # Get all relevant activity files
        for activity_code in self.activity_mapping.keys():
            pattern = os.path.join(subject_path, f"{activity_code}_*.txt")
            files = glob.glob(pattern)
            
            for file_path in files:
                filename = os.path.basename(file_path)
                label = self.get_activity_label(filename)
                
                if label == 'unknown':
                    continue
                
                print(f"Processing {filename} -> {label}")
                data = self.load_file(file_path)
                
                if data is None or len(data) < window_size:
                    continue
                
                # Extract features using sliding window
                extractor = CustomFeatureExtractor()
                
                for start_idx in range(0, len(data) - window_size + 1, overlap):
                    end_idx = start_idx + window_size
                    window_data = data[start_idx:end_idx]
                    
                    features = extractor.extract_all_features(window_data)
                    all_features.append(features)
                    all_labels.append(label)
        
        return all_features, all_labels

def main():
    """Main training function"""
    print("=== Custom Feature Extraction Training ===")
    print("Activities: walking, jogging, sitting, stairs, falls")
    
    # Configuration
    dataset_path = "/Users/amir/Documents/projects/earpiece-fall-detection/DataSet/SisFall_dataset"
    window_size = 400  # 2 seconds at 200Hz
    overlap = 200      # 50% overlap
    
    # Load data from SA01 subject first
    print("\nLoading data from SA01...")
    loader = SisFallDataLoader(dataset_path)
    features_list, labels_list = loader.load_subject_data('SA01', window_size, overlap)
    
    if not features_list:
        print("No data loaded! Check the dataset path and files.")
        return
    
    print(f"Loaded {len(features_list)} feature vectors")
    print(f"Activity distribution:")
    unique_labels, counts = np.unique(labels_list, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['activity'] = labels_list
    
    print(f"\nFeature matrix shape: {features_df.shape}")
    print(f"Number of features: {features_df.shape[1] - 1}")
    
    # Prepare data for training
    X = features_df.drop('activity', axis=1)
    y = features_df['activity']
    
    # Handle any infinite or NaN values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = rf.score(X_train_scaled, y_train)
    test_score = rf.score(X_test_scaled, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Predictions and detailed evaluation
    y_pred = rf.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=rf.classes_, yticklabels=rf.classes_)
    plt.title('Confusion Matrix - Custom Features')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('custom_features_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('custom_features_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model and scaler
    joblib.dump(rf, 'custom_features_model.pkl')
    joblib.dump(scaler, 'custom_features_scaler.pkl')
    
    # Save feature names
    feature_names = list(X.columns)
    joblib.dump(feature_names, 'custom_features_names.pkl')
    
    print("\nModel saved as 'custom_features_model.pkl'")
    print("Scaler saved as 'custom_features_scaler.pkl'")
    print("Feature names saved as 'custom_features_names.pkl'")
    
    # Save feature extraction results
    features_df.to_csv('custom_features_SA01.csv', index=False)
    print("Features saved as 'custom_features_SA01.csv'")

if __name__ == "__main__":
    main()
