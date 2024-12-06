import numpy as np 
from sklearn.preprocessing import StandardScaler


def create_dataset(subject_list, window, overlap, num_columns=16, num_classes=2):

    x_data = np.empty((0, window, num_columns))
    y_data = np.empty((0, 1))  # Labels
    subj_inputs = []  # Tracks number of windows per subject
    
    dataset_dir = '/home/marta/Documenti/eeg_rnn_repo/rnn-eeg-ad/eeg2'
    # print('\n### Creating dataset')
    tot_rows = 0
    
    # for subject_id, category_label in subject_list:
    subject_id = subject_list[0]
    category_label = subject_list[1]
    
    # print(f"aaaaaaaaaaaaaa{subject_id}")
    # print(f"bbbbbbbbbbbbb{category_label}")
    subj_inputs.append(0)  # Initialize window count for this subject
    
    # Load EEG data
    file_path = f"{dataset_dir}/S{subject_id}_{category_label}.npz"
    eeg = np.load(file_path)['eeg'].T  # Transpose if necessary to get [samples, channels]
    
    # Scale EEG data
    scaler = StandardScaler()
    eeg = scaler.fit_transform(eeg)
    
    assert eeg.shape[1] == num_columns, f"Expected {num_columns} channels, got {eeg.shape[1]}"
    
    # Calculate number of windows
    num_windows = 0
    i = 0
    while i + window <= len(eeg):
        i += (window - overlap)
        num_windows += 1
    
    # Preallocate x_data for this subject
    x_data_part = np.empty((num_windows, window, num_columns))
    
    # Extract windows
    i = 0
    for w in range(num_windows):
        x_data_part[w] = eeg[i:i + window]
        i += (window - overlap)
    
    # Update x_data and y_data
    x_data = np.vstack((x_data, x_data_part))
    y_data = np.vstack((y_data, np.full((num_windows, 1), (category_label == 'AD'))))  # Binary label
    subj_inputs[-1] = num_windows
    tot_rows += len(eeg)
    
    # print(f"Total samples: {tot_rows}")
    # print(f"x_data shape: {x_data.shape}")
    # print(f"y_data shape: {y_data.shape}")
    # print(f"Windows per subject: {subj_inputs}")
    # print(f"Class distribution: {[np.sum(y_data == cl) for cl in range(num_classes)]}")
    
    return x_data, y_data, subj_inputs
