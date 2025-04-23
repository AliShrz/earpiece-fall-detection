import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import os

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


# define function for feature extraction Standard deviation magnitude on horizontal plane
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


# define function for feature extraction Standard deviation magnitude
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

# data_array = all_data
# f2 = max_peak_to_peak_amp(data_array)
# f3 = standard_deviation_magnitude_h(data_array)


# crate window for visualizing time of fall

# if value of x, y and z axis is less than 0.5 then window starts
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