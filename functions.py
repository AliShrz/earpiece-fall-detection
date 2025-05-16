import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import os
import zipfile


# calculate fft of signal
def calculate_fft(signal, fs):
    N = len(signal)
    T = 1.0 / fs  # Sampling interval (assuming 50 Hz sampling rate)
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    return xf, 2.0 / N * np.abs(yf[0:N // 2])

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Filter parameters
def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y



# define function for feature extraction (Sum vector magnitude on horizontal plane)
def sum_vector_magnitude_xz(signal_list, method='all'):
    """
    Compute sum vector magnitude feature from a list of sensor data arrays.

    Parameters:
        signal_list: list of np.ndarray, each of shape (6, n_samples)
                        Rows 0-2: Accelerometer [x, y, z]
        method: str, one of ['mean', 'max', 'all']
                - 'mean': return mean of value for the whole signal
                - 'max': return max of value
                - 'all': return list of values for every sample

    Returns:
        List of C2 feature values (one per signal input)
    """
    features = []

    for data in signal_list:
        acc_x = data[0, :]
        acc_z = data[2, :]

        values = np.sqrt((acc_x)**2 + (acc_z)**2)

        if method == 'mean':
            features.append(np.mean(values))
        elif method == 'max':
            features.append(np.max(values))
        elif method == 'all':
            features.append(values)
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'max', or 'all'.")
        
    print(f"Number of features extracted (f1): {len(features)}")
    return features

# define function for feature extraction (Sum vector magnitude on horizontal plane)
def sum_vector_magnitude_xyz(signal_list, method='all'):
    """
    Compute sum vector magnitude feature from a list of sensor data arrays.

    Parameters:
        signal_list: list of np.ndarray, each of shape (6, n_samples)
                        Rows 0-2: Accelerometer [x, y, z]
        method: str, one of ['mean', 'max', 'all']
                - 'mean': return mean of value for the whole signal
                - 'max': return max of value
                - 'all': return list of values for every sample

    Returns:
        List of C2 feature values (one per signal input)
    """
    features = []

    for data in signal_list:
        acc_x = data[0, :]
        acc_y = data[1, :]
        acc_z = data[2, :]

        values = np.sqrt((acc_x)**2 + (acc_y)**2 + (acc_z)**2)

        if method == 'mean':
            features.append(np.mean(values))
        elif method == 'max':
            features.append(np.max(values))
        elif method == 'all':
            features.append(values)
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'max', or 'all'.")
        
    print(f"Number of features extracted (f1): {len(features)}")
    return features

def max_peak_to_peak_amp(signal_list, window_size=50, step_size=25):
    """
    Compute max peak to peak amplitude feature from a list of sensor data arrays.
    
    Parameters:
        signal_list:    list of np.ndarray, each of shape (6, n_samples)
                        Rows 0-2: Accelerometer [x, y, z]
                        Rows 3-5: Gyroscope [x, y, z] (ignored)
        window_size: int, number of samples in each window
        step_size: int, sliding window step size

    Returns:
        List of feature values (one per signal input)
    """
    features = []

    for data in signal_list:
        # Compute acceleration magnitude: √(x² + y² + z²)
        acc_mag = np.linalg.norm(data[0:3, :], axis=0)

        segment_values = []

        # Slide over the signal with the given window size and step size
        for i in range(0, acc_mag.shape[0] - window_size + 1, step_size):
            segment = acc_mag[i:i + window_size]
            
            # Compute max and min for this segment
            max_val = np.max(segment)
            min_val = np.min(segment)
            
            # Calculate the RMS of the range (max - min)
            range_val = max_val - min_val
            rms_c3 = np.sqrt(np.mean(np.square(range_val)))  # RMS of a single value (range)
            
            segment_values.append(rms_c3)

        # Append the list of C3 values for this signal
        features.append(segment_values)

    print(f"Number of features extracted (f2): {len(features)}")
    return features

##########################################################################################

def standard_deviation_magnitude_h(signal_list, window_size=50, step_size=25):
    """
    Compute Standard Deviation Magnitude feature from a list of 2D sensor data arrays.
    
    Parameters:
        signal_list: list of np.ndarray, each of shape (6, n_samples)
                        Rows 0-2: Accelerometer [x, y, z]
                        Rows 3-5: Gyroscope [x, y, z]
        window_size: int, size of the sliding window
        step_size: int, step size of the sliding window
    
    Returns:
        List of C8 values (one per input signal)
    """
    features = []

    for data in signal_list:
        acc_x = data[0, :]
        acc_z = data[2, :]

        values = []

        for i in range(0, acc_x.shape[0] - window_size + 1, step_size):
            segment_x = acc_x[i:i + window_size]
            segment_z = acc_z[i:i + window_size]

            std_x = np.std(segment_x)
            std_z = np.std(segment_z)

            value = np.sqrt(std_x**2 + std_z**2)
            values.append(value)
        
        features.append(values)
        # Optionally: use mean or max of all C8 values in the signal
        # features.append(np.mean(values))  # or np.max(values)

    print(f"Number of features extracted (f3): {len(features)}")
    return features

##########################################################################################

def standard_deviation_magnitude(signal_list, window_size=50, step_size=25):
    """
    Compute Standard Deviation Magnitude feature from a list of 6-row sensor data arrays.
    
    Parameters:
        signal_list: list of np.ndarray, each of shape (6, n_samples)
                        Rows 0-2: Accelerometer [x, y, z]
                        Rows 3-5: Gyroscope [x, y, z] (ignored)
        window_size: int, number of samples in each window
        step_size: int, sliding step size

    Returns:
        List of C9 values (one per signal input)
    """
    features = []

    for data in signal_list:
        acc_x = data[0, :]
        acc_y = data[1, :]
        acc_z = data[2, :]

        values = []

        for i in range(0, acc_x.shape[0] - window_size + 1, step_size):
            seg_x = acc_x[i:i + window_size]
            seg_y = acc_y[i:i + window_size]
            seg_z = acc_z[i:i + window_size]

            std_x = np.std(seg_x)
            std_y = np.std(seg_y)
            std_z = np.std(seg_z)

            value = np.sqrt(std_x**2 + std_y**2 + std_z**2)
            values.append(value)

        # Optionally summarize: mean or max or keep the full list
        # features.append(np.mean(values))  # or np.max(values)
        features.append(values)

    print(f"Number of features extracted (f4): {len(features)}")
    return features

##########################################################################################

def find_fall_window(signal, threshold=600):
    """
    Find the start and end indices of the fall window in a signal.
    
    Parameters:
        signal: np.ndarray, the signal data
        threshold: float, threshold for detecting fall
    
    Returns:
        start: int, start index of the fall window
        end: int, end index of the fall window
    """
    # Find where all three axes are below the threshold
    # mask = (np.abs(signal[0]) < threshold) & (np.abs(signal[1]) < threshold) & (np.abs(signal[2]) < threshold)
    # mask_value = (np.abs(signal[0]) + np.abs(signal[1]) + np.abs(signal[2]))
    mask = (np.abs(signal[0]) + np.abs(signal[2])) > threshold

    # Find the start and end indices of the fall window
    start = np.where(mask)[0][0] if np.any(mask) else None
    # end = np.where(mask)[0][-1] if np.any(mask) else None
    end = start + 100 if start is not None else None  # Assuming a fixed window size of 2000 samples
    # print(mask_value)
    return start, end

##########################################################################################

def plot_fall_window(signal, start, end):
    """
    Plot a window of the fall signal.
    
    Parameters:
        signal: np.ndarray, the signal data
        start: int, start index of the window
        end: int, end index of the window
    """
    plt.figure(figsize=(10, 5))
    if start is not None and end is not None:
        time = np.arange(start, end) / 200  # Assuming a sampling rate of 200 Hz
        plt.plot(time, signal[0, start:end], label='Fall Signal')  # Use the first axis of the signal for plotting
    else:
        print("Invalid start or end indices. Cannot plot the fall window.")
    plt.title('Window of Fall Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

# Make all data the same length by keeping the middle part
# Return a new list with all signals trimmed to the same length (centered)

##########################################################################################

def data_equal_trim(signals):
    min_length = min(matrix.shape[1] for matrix in signals)
    trimmed_signals = []

    for i in range(len(signals)):
        d_length = signals[i].shape[1]
        if d_length > min_length:
            diff = d_length - min_length
            start = diff // 2
            trimmed = signals[i][:, start:start + min_length]
        else:
            trimmed = signals[i].copy()  # keep original if already min length

        trimmed_signals.append(trimmed)
    
    return trimmed_signals

##########################################################################################

def plot_confusion_matrix(cm, classes, normalize = False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:   
        print('Confusion matrix, without normalization')
        print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '0.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

##########################################################################################

def ts_fresh_format(input_data, labels, sample_rate=200):
    """
    Converts raw 6-axis sensor data into a long DataFrame suitable for tsfresh.
    Shows progress from 0% to 100%.
    
    input_data: list or array of shape (num_samples, 6, time_steps)
    labels: list of length num_samples
    """
    all_dfs = []
    total = len(input_data)
    time = np.arange(0, len(input_data[0][0, :])) / sample_rate

    for i in range(total):
        sample = input_data[i]
        df = pd.DataFrame({
            'ID': i + 1,
            'Tag': labels[i],
            'Time': time,
            'Acc X': sample[0, :],
            'Acc Y': sample[1, :],
            'Acc Z': sample[2, :],
            'Gyro X': sample[3, :],
            'Gyro Y': sample[4, :],
            'Gyro Z': sample[5, :]
        })
        all_dfs.append(df)

        # Print progress
        if (i + 1) % (total // 100) == 0 or i == total - 1:  # Every 1%
            percent = int((i + 1) / total * 100)
            print(f"\rProgress: {percent}%", end='')

    print("\n✅ Finished creating the DataFrame.")
    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all

##########################################################################################

def split_signal(signal, length):
    """
    Split a signal into n equal parts.
    """
    signl_length = signal.shape[1]
    n = signl_length // length
    split_signals = []
    start = 0
    end = length
    for i in range(n):
        trimmed = signal[:, start:end]
        start = end
        end += length
        split_signals.append(trimmed)
    print(f"shape: {np.shape(split_signals)}")
    return split_signals

##########################################################################################

def split_and_add(input_data, length, file_names):
    """
    Splits signals and creates file/activity lists (optimized).
    """
    new_data = []
    new_file_names = []
    activity_code = file_names[0].split('_')[0]

    for i in range(len(input_data)):
        file_name = file_names[i]
        signal = input_data[i]
        signl_length = signal.shape[1]
        n = signl_length // length
        start = 0
        end = length
        for _ in range(n):
            trimmed = signal[:, start:end]
            new_data.append(trimmed)
            new_file_names.append(file_name)

    new_activity_code_list = [activity_code] * len(new_data)  # Create new activity code list
    new_data = np.array(new_data)  # Convert to numpy array
    print(f"number of new {activity_code} data: {len(new_data)} with shape: {np.shape(new_data)}, "
          f"file_names :{len(new_file_names)}, activities: {len(new_activity_code_list)} ")

    return new_data, new_file_names, new_activity_code_list

##########################################################################################

def read_zip(zip_file_path, base_path_in_zip, subject_ids ):

    file_name_list = []          # List to store filenames
    all_data = []               # List to store all data (each item is a 2D array from a file)
    all_labels = []             # List to store labels corresponding to each data array
    activity_code_list = []     # List to store activity codes
    adls = []                   # List to store ADL data
    falls = []                  # List to store Fall data

    counter = 0
    ADL = 0
    FALL = 0

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for subject_id in subject_ids:
                folder_path_in_zip = os.path.join(base_path_in_zip, subject_id)

                for item in zip_ref.namelist():
                    if item.startswith(folder_path_in_zip + '/') and item.endswith('.txt'):
                        filename_with_path = item
                        filename = os.path.basename(filename_with_path)

                        try:
                            # Extract activity code from filename (assuming format is like 'D01_01.txt')
                            activity_code = filename.split('_')[0]
                            activity_code_list.append(activity_code)
                            file_name_list.append(filename)

                            # Open the file within the zip and read it with pandas
                            with zip_ref.open(filename_with_path) as file:
                                df = pd.read_csv(file, header=None, delimiter=',', usecols=[0, 1, 2, 3, 4, 5], on_bad_lines='skip')
                                data = df.to_numpy()    # Convert to NumPy array
                                data = data.transpose()  # Transpose the data to get the desired shape

                                # determine label
                                if activity_code.startswith('D'):
                                    adls.append(data)
                                    label = 'ADL'
                                    ADL += 1
                                elif activity_code.startswith('F'):
                                    falls.append(data)
                                    label = 'Fall'
                                    FALL += 1
                                else:
                                    label = 'Unknown'

                                # Append the 2D array to the list
                                all_data.append(data)
                                all_labels.append(label)
                                counter += 1
                                print(f'\rProgress: {counter}/{4505} files processed', end='')

                        except Exception as e:
                            print(f"\nError reading {filename_with_path} from zip: {e}")

    except FileNotFoundError:
        print(f"\nError: Zip file not found at {zip_file_path}")
    except zipfile.BadZipFile:
        print(f"\nError: Invalid zip file at {zip_file_path}")
    except Exception as e:
        print(f"\nError reading {filename_with_path} from zip: {e}")

    print(f"\nTotal files processed: {counter} ✅")
    print(f"Total ADL labels: {ADL} ✅")
    print(f"Total Fall labels: {FALL} ✅")
    return all_data, all_labels, activity_code_list, file_name_list, adls, falls

##########################################################################################

def read_file(base_path, subject_ids):

    file_name_list = []          # List to store filenames
    all_data = []               # List to store all data (each item is a 2D array from a file)
    all_labels = []             # List to store labels corresponding to each data array
    activity_code_list = []     # List to store activity codes
    adls = []                   # List to store ADL data
    falls = []                  # List to store Fall data

    counter = 0
    ADL = 0
    FALL = 0

    for subject_id in subject_ids:
        folder_path = os.path.join(base_path, subject_id)

        if not os.path.isdir(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)

                try:
                    # Extract activity code from filename (assuming format is like 'D01_01.txt')
                    activity_code = filename.split('_')[0]
                    activity_code_list.append(activity_code)
                    file_name_list.append(filename)

                    # Load the first 6 columns of comma-separated file
                    df = pd.read_csv(file_path, header=None, delimiter=',', usecols=[0, 1, 2, 3, 4, 5], on_bad_lines='skip')
                    data = df.to_numpy()    # Convert to NumPy array
                    data = data.transpose()  # Transpose the data to get the desired shape

                    # determine label
                    if activity_code.startswith('D'):
                        adls.append(data)
                        label = 'ADL'
                        ADL += 1
                    elif activity_code.startswith('F'):
                        falls.append(data)
                        label = 'Fall'
                        FALL += 1
                    else:
                        label = 'Unknown'

                    # Append the 2D array to the list
                    all_data.append(data)
                    all_labels.append(label)
                    counter += 1
                    print(f'\rProgress: {counter}/{4505} files processed', end='')


                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"\nTotal files processed: {counter} ✅")
    print(f"Total ADL labels: {ADL} ✅")
    print(f"Total Fall labels: {FALL} ✅")
    return all_data, all_labels, activity_code_list, file_name_list, adls, falls

##########################################################################################

def idle_remover(data_list, window_size, scale, mode):    # modes: 'acc', 'gyro', 'both'

    all_cleaned_data = []

    for i in range(len(data_list)):
        signal = data_list[i]  # shape: (6, N)
        acc = signal[0:3]
        gyro = signal[3:6]

        # Precompute thresholds
        acc_combined = np.abs(acc[0]) + np.abs(acc[1]) + np.abs(acc[2])
        gyro_combined = np.abs(gyro[0]) + np.abs(gyro[1]) + np.abs(gyro[2])

        acc_threshold = np.var(acc_combined) / scale
        gyro_threshold = np.var(gyro_combined) / scale

        windowed_data = []

        for j in range(0, signal.shape[1] - window_size + 1, window_size):
            acc_window = acc_combined[j:j + window_size]
            gyro_window = gyro_combined[j:j + window_size]

            acc_var = np.var(acc_window)
            gyro_var = np.var(gyro_window)

            # Check condition based on selected mode
            keep = False
            if mode == 'acc':
                keep = acc_var >= acc_threshold
            elif mode == 'gyro':
                keep = gyro_var >= gyro_threshold
            elif mode == 'both':
                keep = (acc_var >= acc_threshold) and (gyro_var >= gyro_threshold)
            else:
                raise ValueError("Invalid mode. Use 'acc', 'gyro', or 'both'.")

            if keep:
                windowed_data.append(signal[:, j:j + window_size])

        if windowed_data:
            cleaned_signal = np.concatenate(windowed_data, axis=1)
        else:
            cleaned_signal = np.empty((6, 0))

        print(f"Signal {i+1}: Original shape = {signal.shape}, Cleaned shape = {cleaned_signal.shape}")
        all_cleaned_data.append(cleaned_signal)

    return all_cleaned_data

##########################################################################################

def plot_signals(data, data_2, title_1, title_2, activity_mapping):
    
    plt.figure(figsize=(15, 10))

    # Plot original signal (first 3 axes)
    plt.subplot(4, 1, 1)
    plt.plot(data[0, :], label='X-axis')
    plt.plot(data[1, :], label='Y-axis')
    plt.plot(data[2, :], label='Z-axis')
    plt.title(f'{activity_mapping[title_1]} - (Accelerometer)')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()

    # Plot cleaned signal (first 3 axes)
    plt.subplot(4, 1, 2)
    plt.plot(data_2[0, :], label='X-axis')
    plt.plot(data_2[1, :], label='Y-axis')
    plt.plot(data_2[2, :], label='Z-axis')
    plt.title(f'{activity_mapping[title_2]} - (Accelerometer)')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()

    # Plot original signal (last 3 axes)
    plt.subplot(4, 1, 3)
    plt.plot(data[3, :], label='X-axis')
    plt.plot(data[4, :], label='Y-axis')
    plt.plot(data[5, :], label='Z-axis')
    plt.title(f'{activity_mapping[title_1]} - (Gyro)')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()

    # Plot cleaned signal (last 3 axes)
    plt.subplot(4, 1, 4)
    plt.plot(data_2[3, :], label='X-axis')
    plt.plot(data_2[4, :], label='Y-axis')
    plt.plot(data_2[5, :], label='Z-axis')
    plt.title(f'{activity_mapping[title_2]} - (Gyro)')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()

    
    plt.tight_layout()
    plt.show()

##########################################################################################

def plot_captured_signals(data, data_1, data_3, title, title_1, title_2, activity_mapping):
    plt.figure(figsize=(15, 12))

    # Row 1: Original Accelerometer
    plt.subplot(4, 1, 1)
    plt.plot(data[0, :], label='X-axis')
    plt.plot(data[1, :], label='Y-axis')
    plt.plot(data[2, :], label='Z-axis')
    plt.title(f'{activity_mapping[title]} - Original Accelerometer')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration')
    plt.grid()
    plt.legend()

    # Row 2: Cleaned Accelerometer 1 (data_1)
    plt.subplot(4, 2, 3)
    plt.plot(data_1[0, :], label='X-axis')
    plt.plot(data_1[1, :], label='Y-axis')
    plt.plot(data_1[2, :], label='Z-axis')
    plt.title(f'{activity_mapping[title_1]} - Cleaned Accelerometer')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration')
    plt.grid()
    plt.legend()

    # Row 2: Cleaned Accelerometer 2 (data_3)
    plt.subplot(4, 2, 4)
    plt.plot(data_3[0, :], label='X-axis')
    plt.plot(data_3[1, :], label='Y-axis')
    plt.plot(data_3[2, :], label='Z-axis')
    plt.title(f'{activity_mapping[title_2]} - Cleaned Accelerometer')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration')
    plt.grid()
    plt.legend()

    # Row 3: Original Gyroscope
    plt.subplot(4, 1, 3)
    plt.plot(data[3, :], label='X-axis')
    plt.plot(data[4, :], label='Y-axis')
    plt.plot(data[5, :], label='Z-axis')
    plt.title(f'{activity_mapping[title]} - Original Gyroscope')
    plt.xlabel('Sample')
    plt.ylabel('Rotation')
    plt.grid()
    plt.legend()

    # Row 4: Cleaned Gyroscope 1 (data_1)
    plt.subplot(4, 2, 7)
    plt.plot(data_1[3, :], label='X-axis')
    plt.plot(data_1[4, :], label='Y-axis')
    plt.plot(data_1[5, :], label='Z-axis')
    plt.title(f'{activity_mapping[title_1]} - Cleaned Gyroscope')
    plt.xlabel('Sample')
    plt.ylabel('Rotation')
    plt.grid()
    plt.legend()

    # Row 4: Cleaned Gyroscope 2 (data_3)
    plt.subplot(4, 2, 8)
    plt.plot(data_3[3, :], label='X-axis')
    plt.plot(data_3[4, :], label='Y-axis')
    plt.plot(data_3[5, :], label='Z-axis')
    plt.title(f'{activity_mapping[title_2]} - Cleaned Gyroscope')
    plt.xlabel('Sample')
    plt.ylabel('Rotation')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

##########################################################################################

def plot_captured_signals(data, data_1, data_3, title, title_1, title_2, activity_mapping):
    plt.figure(figsize=(15, 12))

    # Row 1: Original Accelerometer
    plt.subplot(4, 1, 1)
    plt.plot(data[0, :], label='X-axis')
    plt.plot(data[1, :], label='Y-axis')
    plt.plot(data[2, :], label='Z-axis')
    plt.title(f'{title} - Original Accelerometer')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration')
    plt.grid()
    plt.legend()

    # Row 2: Cleaned Accelerometer 1 (data_1)
    plt.subplot(4, 2, 3)
    plt.plot(data_1[0, :], label='X-axis')
    plt.plot(data_1[1, :], label='Y-axis')
    plt.plot(data_1[2, :], label='Z-axis')
    plt.title(f'{title_1} - first part Accelerometer')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration')
    plt.grid()
    plt.legend()

    # Row 2: Cleaned Accelerometer 2 (data_3)
    plt.subplot(4, 2, 4)
    plt.plot(data_3[0, :], label='X-axis')
    plt.plot(data_3[1, :], label='Y-axis')
    plt.plot(data_3[2, :], label='Z-axis')
    plt.title(f'{title_2} - second part Accelerometer')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration')
    plt.grid()
    plt.legend()

    # Row 3: Original Gyroscope
    plt.subplot(4, 1, 3)
    plt.plot(data[3, :], label='X-axis')
    plt.plot(data[4, :], label='Y-axis')
    plt.plot(data[5, :], label='Z-axis')
    plt.title(f'{title} - Original Gyroscope')
    plt.xlabel('Sample')
    plt.ylabel('Rotation')
    plt.grid()
    plt.legend()

    # Row 4: Cleaned Gyroscope 1 (data_1)
    plt.subplot(4, 2, 7)
    plt.plot(data_1[3, :], label='X-axis')
    plt.plot(data_1[4, :], label='Y-axis')
    plt.plot(data_1[5, :], label='Z-axis')
    plt.title(f'{title_1} - first part Gyroscope')
    plt.xlabel('Sample')
    plt.ylabel('Rotation')
    plt.grid()
    plt.legend()

    # Row 4: Cleaned Gyroscope 2 (data_3)
    plt.subplot(4, 2, 8)
    plt.plot(data_3[3, :], label='X-axis')
    plt.plot(data_3[4, :], label='Y-axis')
    plt.plot(data_3[5, :], label='Z-axis')
    plt.title(f'{title_2} - second part Gyroscope')
    plt.xlabel('Sample')
    plt.ylabel('Rotation')
    plt.grid()
    plt.legend()

    plt.suptitle(f'Comparison of Original and Cleaned Signals for {activity_mapping[title]}', fontsize=16)
    plt.tight_layout()
    plt.show()

##########################################################################################

def keep_from_peak(data_list, window_size):
    all_cleaned_data = []
    
    for i, signal in enumerate(data_list):
        # Compute the combined absolute magnitude of the first 3 axes
        combined = np.abs(signal[0]) + np.abs(signal[1]) + np.abs(signal[2])
        
        # Find the index of the peak
        peak_index = np.argmax(combined)
        
        # Determine start and end of window
        start = max(0, peak_index - window_size)
        end = min(signal.shape[1], peak_index + window_size)

        # Slice the signal and store it
        cleaned = signal[:, start:end]
        all_cleaned_data.append(cleaned)
        
        print(f"Signal {i+1}: Original shape = {signal.shape}, Kept shape = {cleaned.shape}")
    
    return all_cleaned_data