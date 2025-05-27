import pandas as pd
import numpy as np
import os
import glob
from typing import Tuple, List
from datetime import datetime, timedelta
import csv
import re
import math
import shutil

class UMADataset:
    """Dataset loader for UMA Fall Detection Dataset"""
    
    def __init__(self, data_path="/DataSet/UMAFall_Dataset/"):
        self.data_path = data_path
        self.dataset_name = "UMAFall_Dataset"
    
    def load(self, clip=False, min_duration=12):
        """
        Load UMA dataset from CSV files
        
        Parameters:
        - clip: Whether to clip acceleration values (similar to sisfall)
        - min_duration: Minimum duration in seconds to keep samples
        
        Returns:
        - DataFrame with columns: SubjectID, Activity, Duration (s), Acc, Target, accel_g
        """
        print("Loading UMA dataset...")
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(self.data_path, "UMAFall_*.csv"))
        print(f"Found {len(csv_files)} CSV files")
        
        if len(csv_files) == 0:
            print("No CSV files found. Please check the path.")
            return pd.DataFrame()
        
        data_list = []
        
        for file_path in csv_files:
            try:
                # Parse filename to extract metadata
                filename = os.path.basename(file_path)
                metadata = self._parse_filename(filename)
                
                if metadata is None:
                    print(f"Skipping file with invalid name: {filename}")
                    continue
                
                # Read and process the file
                df_file = self._read_uma_file(file_path)
                
                if df_file is not None and len(df_file) > 0:
                    # Calculate duration
                    duration = (df_file['TimeStamp'].max() - df_file['TimeStamp'].min()) / 1000.0  # Convert to seconds
                    
                    # Skip if too short
                    if duration < min_duration:
                        continue
                    
                    # Process acceleration data
                    acc_data = df_file[['X-Axis', 'Y-Axis', 'Z-Axis']].values
                    
                    # Convert to similar format as sisfall (multiply by ~1000 to match scale)
                    acc_data_scaled = acc_data * 1000  # Convert from g to similar scale as sisfall
                    
                    # Calculate magnitude
                    accel_g = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in acc_data_scaled]
                    
                    if clip:
                        accel_g = self._clip_arr(accel_g)
                    
                    # Create row for this file
                    row = {
                        'SubjectID': metadata['subject_id'],
                        'Activity': metadata['activity'],
                        'Duration (s)': duration,
                        'Acc': acc_data_scaled.tolist(),  # Keep original format
                        'Target': 1 if metadata['is_fall'] else 0,  # 1 for fall, 0 for ADL
                        'accel_g': accel_g
                    }
                    
                    data_list.append(row)
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        if not data_list:
            print("No valid data found!")
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list)
        print(f"Loaded {len(df)} samples from {df['SubjectID'].nunique()} subjects")
        print(f"Falls: {df['Target'].sum()}, ADLs: {(df['Target'] == 0).sum()}")
        
        return df
    
    def _parse_filename(self, filename):
        """Parse UMA filename to extract metadata"""
        # Remove extension
        name = filename.replace('.csv', '')
        
        # Pattern: UMAFall_Subject_XX_[ADL/Fall]_ActivityName_TrialNo_DateTime
        pattern = r'UMAFall_Subject_(\d+)_(ADL|Fall)_([^_]+)_(\d+)_(.+)'
        match = re.match(pattern, name)
        
        if match:
            subject_id = f"S{match.group(1).zfill(2)}"  # Format as S01, S02, etc.
            movement_type = match.group(2)
            activity_name = match.group(3)
            trial_no = match.group(4)
            datetime_str = match.group(5)
            
            return {
                'subject_id': subject_id,
                'is_fall': movement_type == 'Fall',
                'activity': f"{movement_type}_{activity_name}",
                'trial_no': trial_no,
                'datetime': datetime_str
            }
        
        return None
    
    def _read_uma_file(self, file_path):
        """Read UMA CSV file, skipping header comments"""
        try:
            # Read file and find where data starts
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the line with column headers (contains "TimeStamp")
            data_start = 0
            for i, line in enumerate(lines):
                if 'TimeStamp' in line and not line.startswith('%'):
                    data_start = i
                    break
            
            # Read the data part
            if data_start < len(lines) - 1:
                # Read from the line after headers
                df = pd.read_csv(file_path, skiprows=data_start+1, header=None, sep=';')
                
                # Set column names
                df.columns = ['TimeStamp', 'Sample No', 'X-Axis', 'Y-Axis', 'Z-Axis', 
                             'Sensor Type', 'Sensor ID']
                
                # Filter for accelerometer data only (Sensor Type = 0)
                df = df[df['Sensor Type'] == 0].reset_index(drop=True)
                
                return df
            
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return None
        
        return None
    
    def _clip_arr(self, arr, clip_value=3.0):
        """Clip acceleration values (similar to sisfall)"""
        return np.clip(arr, -clip_value, clip_value)
    
    def get_X_y(self, df):
        """Extract features and labels (compatible with sisfall format)"""
        # This should match the feature extraction used in your sisfall models
        # For now, using the magnitude as a simple feature
        
        X = []
        y = []
        
        for _, row in df.iterrows():
            # Use the magnitude values as features
            accel_g = row['accel_g']
            
            # Create features similar to sisfall
            # You might need to adjust this based on your actual feature extraction
            if len(accel_g) > 0:
                features = [
                    np.mean(accel_g),
                    np.std(accel_g),
                    np.max(accel_g),
                    np.min(accel_g),
                    np.percentile(accel_g, 25),
                    np.percentile(accel_g, 75),
                    len(accel_g)  # sequence length
                ]
                
                X.append(features)
                y.append(row['Target'])
        
        return np.array(X), np.array(y)

def test_models_on_uma(trained_models, uma_df, sisfall_dataset):
    """
    Test trained models on UMA dataset
    
    Parameters:
    - trained_models: Dictionary of trained models from sisfall
    - uma_df: UMA dataset DataFrame
    - sisfall_dataset: Original sisfall dataset object for feature extraction
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("=== TESTING MODELS ON UMA DATASET ===")
    
    # Create UMA dataset object for feature extraction
    uma_dataset = UMADataset()
    
    # Extract features from UMA data
    X_uma, y_uma = uma_dataset.get_X_y(uma_df)
    
    print(f"UMA test set: X: {X_uma.shape}, y: {y_uma.shape}")
    print(f"UMA distribution ([ADLs, Falls]): {np.bincount(y_uma)}")
    
    # Test each model
    results = {}
    
    for model_name, model_info in trained_models.items():
        print(f"\n--- Testing {model_name} ---")
        
        model = model_info['model']
        
        try:
            # Make predictions
            y_pred = model.predict(X_uma)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_uma, y_pred)
            precision = precision_score(y_uma, y_pred)
            recall = recall_score(y_uma, y_pred)
            f1 = f1_score(y_uma, y_pred)
            
            results[model_name] = {
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            results[model_name] = None
    
    # Plot confusion matrices
    valid_models = {k: v for k, v in results.items() if v is not None}
    
    if valid_models:
        fig, axes = plt.subplots(1, len(valid_models), figsize=(6*len(valid_models), 5))
        if len(valid_models) == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(valid_models.items()):
            cm = confusion_matrix(y_uma, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx],
                       xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall'])
            axes[idx].set_title(f'{model_name}\nUMA Dataset')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed classification reports
        print("\n=== DETAILED CLASSIFICATION REPORTS ON UMA ===")
        for model_name, result in valid_models.items():
            if result is not None:
                print(f"\n{model_name} - UMA Dataset Classification Report:")
                print(classification_report(y_uma, result['predictions'], 
                                          target_names=['ADL', 'Fall']))
    
    return results


def transform_uma_to_sisfall_format(uma_files, activity_mapping=None):
    """
    Transform UMA dataset to match SisFall format shown in the picture
    
    Parameters:
    - uma_files: list of UMA CSV file paths
    - activity_mapping: dict to map activity numbers to names (optional)
    
    Returns:
    - transformed_df: DataFrame in SisFall format
    """
    
    if activity_mapping is None:
        activity_mapping = {
            1: "Activity1",
            # Add more mappings as needed
        }
    
    transformed_rows = []
    
    for file_idx, file_path in enumerate(uma_files):        
        # Load UMA file
        uma_df = pd.read_csv(file_path)
        
        # Check required columns
        accel_cols = ['Accelerometer: x-axis (g)', 'Accelerometer: y-axis (g)', 'Accelerometer: z-axis (g)']
        if not all(col in uma_df.columns for col in accel_cols):
            print(f"Warning: Missing accelerometer columns in {file_path}")
            continue
        
        # Group by Subject, Activity, and Trial (if available)
        group_cols = []
        if 'Subject' in uma_df.columns:
            group_cols.append('Subject')
        if 'Activity' in uma_df.columns:
            group_cols.append('Activity')
        if 'Trial' in uma_df.columns:
            group_cols.append('Trial')
        
        if not group_cols:
            # If no grouping columns, treat entire file as one activity
            group_cols = ['dummy_group']
            uma_df['dummy_group'] = 1
        
        # Group the data
        grouped = uma_df.groupby(group_cols)
        
        for group_name, group_data in grouped:
            # Extract accelerometer data
            acc_data = group_data[accel_cols].values.tolist()
            
            # Calculate duration (assuming consistent sampling rate)
            if 'TimeStamp' in group_data.columns:
                timestamps = group_data['TimeStamp'].values
                duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
            else:
                # Estimate duration based on number of samples (assuming ~50Hz sampling)
                duration = len(acc_data) / 50.0  # Adjust sampling rate as needed
            
            # Determine subject ID
            if isinstance(group_name, tuple) and 'Subject' in group_cols:
                subject_idx = group_cols.index('Subject')
                subject_id = f"SA{group_name[subject_idx]:02d}"
            else:
                subject_id = f"SA{file_idx + 1:02d}"
            
            # Determine activity name
            if isinstance(group_name, tuple) and 'Activity' in group_cols:
                activity_idx = group_cols.index('Activity')
                activity_num = group_name[activity_idx]
                activity_name = activity_mapping.get(activity_num, f"Activity{activity_num}")
            else:
                activity_name = "Activity1"
            
            # Determine target (0 for ADL, 1 for Fall)
            # You can adjust this logic based on your UMA dataset labeling
            if 'Tag' in group_data.columns:
                # Use the most common tag in the group
                target = group_data['Tag'].mode().iloc[0] if not group_data['Tag'].mode().empty else 0
            else:
                # Default to ADL (0) - adjust as needed
                target = 0
            
            # Create the row in SisFall format
            transformed_row = {
                'SubjectID': subject_id,
                'Activity': activity_name,
                'Duration (s)': round(duration, 1),
                'Acc': acc_data,
                'Target': target,
                'accel_g': acc_data.copy()  # Same as Acc for compatibility
            }
            
            transformed_rows.append(transformed_row)
    
    # Create DataFrame
    transformed_df = pd.DataFrame(transformed_rows)
    return transformed_df
    
    
# Usage example:
def run_uma_evaluation(sisfall_dataset, sisfall_df, trained_models):
    """Complete pipeline to load UMA and test models"""
    
    # Load UMA dataset
    uma_dataset = UMADataset()
    uma_df = uma_dataset.load(clip=True, min_duration=12)
    
    if uma_df.empty:
        print("Failed to load UMA dataset!")
        return None
    
    # Test models on UMA
    uma_results = test_models_on_uma(trained_models, uma_df, sisfall_dataset)
    
    return uma_results, uma_df

def uma_search_csv_files(directory, activities_of_interest=None):
    csv_files = []
    for current_folder, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                if activities_of_interest is not None:
                    for activity in activities_of_interest:
                        if activity in file:
                            full_path = os.path.join(current_folder, file)
                            csv_files.append(full_path)
                            break
                else:
                    full_path = os.path.join(current_folder, file)
                    csv_files.append(full_path)
                    break
    return csv_files


def Downsampled(diretorio_raiz, old_freq, new_freq):
        
        taxa_amostragem_original = old_freq  # Hz
        taxa_amostragem_desejada = new_freq  # Hz
        # Desired downsampling factor
        fator_downsampling = int(taxa_amostragem_original/taxa_amostragem_desejada)
        
            
        # Traverse all subfolders and CSV files
        for pasta_atual, subpastas, arquivos in os.walk(diretorio_raiz):
            for arquivo_entrada in arquivos:
                if arquivo_entrada.endswith('.csv'):
                    caminho_arquivo = os.path.join(pasta_atual, arquivo_entrada)
            
                    
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(caminho_arquivo)
        
                    # Extract the header from the original file
                    cabeçalho = list(df.columns)
        
                    # Select only the columns of interest for downsampling
                    colunas_interesse = cabeçalho[1:7]
        
                    # Perform downsampling on the data
                    dados_downsampled = df[colunas_interesse].iloc[::fator_downsampling]
                    tempo_downsampled = df['TimeStamp'].iloc[::fator_downsampling]
                    info_downsampled = df.iloc[::fator_downsampling, 7:]
        
                    # Create a new DataFrame with the downsampled data and original information
                    df_downsampled = pd.concat([tempo_downsampled, dados_downsampled, info_downsampled], axis=1)
        
                    # Create the destination path for the new CSV file
                    nome_arquivo = os.path.splitext(arquivo_entrada)[0]
                    caminho_arquivo_downsampled = os.path.join(pasta_atual, nome_arquivo + '.csv')
        
                    # Save the downsampled data to a new CSV file, keeping the original header
                    df_downsampled.to_csv(caminho_arquivo_downsampled, index=False, header=cabeçalho)




def format_timestamp(timestamp):
    
    if timestamp == 'TimeStamp':
        return timestamp

    date_time = datetime.strptime(timestamp, "%Y-%m-%d %H-%M-%S")
    formatted_date_time = date_time.strftime("%Y/%m/%dT%H:%M:%S")

    return formatted_date_time

def add_milliseconds(timestamp, milliseconds):

    # Convert the time string to a datetime object
    time_object = datetime.strptime(timestamp[11:], '%H:%M:%S')
    
    new_time = time_object + timedelta(milliseconds=int(milliseconds))
    
    return new_time.strftime(timestamp[:11] + '%H:%M:%S.%f')


def process_to_up(output_folder):
    # Root directory where the CSV files are located
    root_directory = output_folder
    
    # Traverse all CSV files in the root directory
    for file in os.listdir(root_directory):
        if file.endswith('.csv'):
            # Extract information from the file name
            file_name = os.path.splitext(file)[0]
            name_parts = file_name.split('_')
            subject = name_parts[1] + name_parts[2].lstrip('0')  # Remove leading zero
            
            activity = None
            activity_name = name_parts[4]
            
            if activity_name == 'Walking':
                activity = 1
            elif activity_name == 'Jogging':
                activity = 2
            elif activity_name == 'Bending':
                activity = 3
            elif activity_name == 'Hopping':
                activity = 4
            elif activity_name == 'GoDownstairs':
                activity = 5
            elif activity_name == 'GoUpstairs':
                activity = 6
            elif activity_name == 'LyingDown' and name_parts[5] == 'OnABed':
                activity = 7
            elif activity_name == 'Sitting' and name_parts[5] == 'GettingUpOnAChair':
                activity = 8
            elif activity_name == 'Aplausing':
                activity = 9
            elif activity_name == 'HandsUp':
                activity = 10
            elif activity_name == 'MakingACall':
                activity = 11
            elif activity_name == 'OpeningDoor':
                activity = 12
            elif activity_name == 'backwardFall':
                activity = 13
            elif activity_name == 'forwardFall':
                activity = 14
            elif activity_name == 'lateralFall':
                activity = 15
            
            # Count the number of occurrences of the "_" character
            num_underscores = file_name.count("_")
            
            # Condition based on the number of "_"
            if num_underscores == 7:
                trial = name_parts[5]
                
            else:
                trial = name_parts[6]
            
            
            
    
            # Create the directory for the subject, activity, and trial
            destination_directory = os.path.join(root_directory, f'{subject}', f'Activity{activity}', f'Trial{trial}')
            os.makedirs(destination_directory, exist_ok=True)
            
            new_name = f'UMAFALL_{subject}Activity{activity}Trial{trial}.csv'
    
            # Move the file to the destination directory with the new name
            shutil.move(os.path.join(root_directory, file), os.path.join(destination_directory, new_name))


def process_UMA(input_folder, output_folder):
    # List all CSV files in the input folder
    csv_files = glob.glob(input_folder + "/*.csv")
    
    
    # Iterate over each CSV file
    for input_file in csv_files:
        # Extract the file name without the extension
        file_name = os.path.splitext(os.path.basename(input_file))[0]
    
        # Extract the subject from the file name
        #subject = file_name.split('_')[0] + " " + file_name.split('_')[2]
        subject = file_name.split('_')[2].lstrip('0')
        # Count the number of occurrences of the "_" character
        num_underscores = file_name.count("_")
        
        # Condition based on the number of "_"
        if num_underscores == 7:
            timeS = file_name.split('_')[6] + " " + file_name.split('_')[7]
            trial = file_name.split('_')[5] 
            formatted_timestamp = format_timestamp(timeS)
            
        else:
            timeS = file_name.split('_')[7] + " " + file_name.split('_')[8]
            trial = file_name.split('_')[6]        
            formatted_timestamp = format_timestamp(timeS)            
              
        # Extract the activity from the file name
        if file_name.split('_')[3] == 'Fall':
            tag = 0
            
        else:
            tag = 0
        
       
        activity = None
        file_info = file_name.split('_')
        activity_name = file_info[4]
        
        if activity_name == 'Walking':
            activity = 1
        elif activity_name == 'Jogging':
            activity = 2
        elif activity_name == 'Bending':
            activity = 3
        elif activity_name == 'Hopping':
            activity = 4
        elif activity_name == 'GoDownstairs':
            activity = 5
        elif activity_name == 'GoUpstairs':
            activity = 6
        elif activity_name == 'LyingDown' and file_info[5] == 'OnABed':
            activity = 7
        elif activity_name == 'Sitting' and file_info[5] == 'GettingUpOnAChair':
            activity = 8
        elif activity_name == 'Aplausing':
            activity = 9
        elif activity_name == 'HandsUp':
            activity = 10
        elif activity_name == 'MakingACall':
            activity = 11
        elif activity_name == 'OpeningDoor':
            activity = 12
        elif activity_name == 'backwardFall':
            activity = 13
        elif activity_name == 'forwardFall':
            activity = 14
        elif activity_name == 'lateralFall':
            activity = 15
            
                    
        # Define the output path for the selected file
        output_file = os.path.join(output_folder, file_name + ".csv")
        
        
        # Initialize the list to store the selected lines
        selected_lines = []
        selected_lines_gyro = []
        selected_lines_acc = []
    
        # Read the CSV file into a list of lines
        with open(input_file, 'r') as file:
            csv_reader = csv.reader(file)
            lines = list(csv_reader)
    
            # Create a new header with the existing columns plus the "Subject" and "Trial" columns
            new_header = ['TimeStamp', 'Accelerometer: x-axis (g)', 'Accelerometer: y-axis (g)', 'Accelerometer: z-axis (g)', 
                          'Gyroscope: x-axis (rad/s)','Gyroscope: y-axis (rad/s)','Gyroscope: z-axis (rad/s)','Subject', 'Activity','Trial', 'Tag']
                            
                            
            for i in range(41, len(lines)):
                line = lines[i][0].split(';')
                if len(line) >= 7:  # Check if the line has enough elements
                    sensor_type = line[5]
                    sensor_id = line[6]
                    if sensor_type.isdigit() and sensor_id.isdigit():
                        sensor_type = int(sensor_type)
                        sensor_id = int(sensor_id)
                        if (sensor_type == 0 and sensor_id == 3):
                            line = line[2:5] # Update the values of "Subject", "Trial" and "Tag"
                            selected_lines_acc.append(line)
                        elif (sensor_type == 1 and sensor_id == 3):
                            #timeStamp = add_milliseconds(formatted_timestamp, line[0])
                            line[0] = int(line[0]) /1000 # convert time from milliseconds to seconds
                            line = line[:1] + line[2:5] + [subject, activity, trial, tag]  # Update the values of "Subject", "Trial" and "Tag"
                            selected_lines_gyro.append(line)
                
        
            
        
        for i in range(len(selected_lines_acc)):
            line_gyro = selected_lines_gyro[i]
            line_acc = selected_lines_acc[i]
            new_line = line_gyro[:1] + line_acc[0:] + line_gyro[1:]
            selected_lines.append(new_line)
        
        
        
        for selected_line in selected_lines[0:]:
            for i in range(4, 7):
                value_degrees_s = selected_line[i]   
                try:
                    value_degrees_s = float(value_degrees_s)
                    value_radians_s = value_degrees_s*(math.pi/180)
                except ValueError:
                    value_radians_s = value_degrees_s  # Assign NaN to the invalid value        
                
                selected_line[i] = value_radians_s
            
          
        
        # Save the selected lines to a new CSV file
        with open(output_file, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            # Write the new header
            csv_writer.writerow(new_header)
            csv_writer.writerows(selected_lines)
            
    
    Downsampled(output_folder,20,18)
    print("Downsample complete")
    process_to_up(output_folder)
    print("Processing completed.") 
   