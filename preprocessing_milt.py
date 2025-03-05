import os

path = "/home/marta/Documenti/eeg-ml-thesis/"
os.chdir(path)

import torch 
torch.set_num_threads(4) 

import torch.nn as nn 
import argparse
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import r_pca 
import scipy.io
from tqdm import tqdm
import datetime 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
import pandas as pd 
import csv

def pca_reduction(A, tol, comp = 0):
  rpca = False
  rpca_mu = 0
  multiscale_pca = False

  assert(len(A.shape) == 2)
  dmin = min(A.shape)
  if rpca:
    r = r_pca.R_pca(A, mu = rpca_mu)
    print('Auto tol:', 1e-7 * r.frobenius_norm(r.D), 'used tol:', tol)
    print('mu', r.mu, 'lambda', r.lmbda)
    L, S = r.fit(tol = tol, max_iter = 10, iter_print = 1)
    global norm_s
    norm_s = np.linalg.norm(S, ord='fro')  # for debug
    print('||A,L,S||:', np.linalg.norm(A, ord='fro'), np.linalg.norm(L, ord='fro'), np.linalg.norm(S, ord='fro'))
    #np.savez_compressed('rpca.npz', pre = A, post = L)
  elif multiscale_pca:
    print('MSPCA...')
    #ms = mspca.MultiscalePCA()
    #L = ms.fit_transform(A, wavelet_func='sym4', threshold=0.1, scale = True )
    print('saving MAT file and calling Matlab...')
    scipy.io.savemat('mspca.mat', {'A': A}, do_compression = True)
    os.system('matlab -batch "mspca(\'mspca.mat\')"')
    L = scipy.io.loadmat('mspca.mat')['L'] 
  else:
    
    L = A
  U, lam, V = np.linalg.svd(L, full_matrices = False)  # V is transposed
  assert(U.shape == (A.shape[0], dmin) and lam.shape == (dmin,) and V.shape == (dmin, A.shape[1]))
  #np.savetxt('singular_values.csv', lam)
  lam_trunc = lam[lam > 0.015 * lam[0]]  # magic number
  p = comp if comp else len(lam_trunc)
  assert(p <= dmin)
  print('PCA truncation', dmin, '->', p)
  return L, V.T[:,:p]


def precompute_crops(subject_list, 
                     window, 
                     overlap, 
                     DATASET_DIR, 
                     save_dir, 
                     csv_file_name,
                     task):

    base_dir = "/home/marta/Documenti/eeg-ml-thesis/"


    
    os.makedirs(save_dir, exist_ok=True)

    task_class = class_groups[task]
    label_mapping = {cls: i for i, cls in enumerate(task_class)}

    print(f"Class Mapping: {label_mapping}")
    mapping_file = os.path.join(base_dir, "output-milt", f"class_mapping_{task}.txt")

    with open(mapping_file, "w") as f:
        f.write(f"Task: {task}\n")
        f.write(f"Class mapping: {label_mapping}\n")

    
    all_crops = []

    for subject_id, category_label in subject_list:
        file_path = f"{DATASET_DIR}/{category_label}/{subject_id}.npy"

        eeg = np.load(file_path).T
        scaler = StandardScaler()
        eeg = scaler.fit_transform(eeg)

        num_columns = eeg.shape[1]
        num_windows = (len(eeg) - window) // (window - overlap) + 1

        i = 0
        for w in range(num_windows):
            x_data = eeg[i:i + window]
            i += (window - overlap)

            crop_filename = f"{subject_id}_{category_label}_crop{w}.npz"
            crop_save_path = os.path.join(save_dir, crop_filename)

            y_label = label_mapping[category_label]
            y_data = np.array([[y_label]])

            np.savez(crop_save_path, x_data=x_data, y_data=y_data)

            all_crops.append((subject_id, w, category_label,y_label, crop_save_path))

    train_ind, val_ind = train_test_split(all_crops, train_size=0.75, random_state=42, shuffle=True)

    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["subject", "crops_number", "category_label", "label", "split", "file_path"])

        for crop in train_ind:
            writer.writerow([crop[0], crop[1], crop[2], crop[3], "train", crop[4]])
        for crop in val_ind:
            writer.writerow([crop[0], crop[1], crop[2], crop[3], "val", crop[4]])

def reduce_matrix(A, V, PCA_COMPONENTS):
    # Check the shape of A
    if len(A.shape) == 2:
        # If A is 2D (w, 16), expand it to (1, w, 16)
        A = np.expand_dims(A, axis=0)

    # Now A should be 3D: (N, w, 16)
    B = np.swapaxes(A, 1, 2)  # Swap axes: (N, 16, w)
    C = B.reshape((-1, B.shape[2]))  # Flatten: ((N*16), w)

    if V is None:
        L, V = pca_reduction(C, 5e-6, comp=PCA_COMPONENTS)

    B = C @ V  # Apply PCA: ((N*16), p)
    B = B.reshape((A.shape[0], A.shape[2], B.shape[1]))  # Reshape: (N, 16, p)

    return np.swapaxes(B, 1, 2), V  # Return: (N, p, 16)


def compute_vpca(csv_file, output_dir, PCA_COMPONENTS):

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    data_info = pd.read_csv(csv_file)

    # Select a random subset of the training data
    train_rows = data_info[data_info["split"] == "train"]
    sampled_train_rows = train_rows.sample(frac=0.2, random_state=42)  # Sample randomly

    train_data = []
    
    print(f"Processing training data for PCA computation on {len(sampled_train_rows)}")
    for _, row in tqdm(sampled_train_rows.iterrows(), 
                       total=len(sampled_train_rows), 
                       desc="Sampling Training Data"):
        npz_data = np.load(row["file_path"])
        x_data = npz_data["x_data"]

        if len(x_data.shape) == 2:  
            x_data = np.expand_dims(x_data, axis=0)  # Convert (w, 16) → (1, w, 16)

        x_data = np.swapaxes(x_data, 1, 2)  # (N, 16, w) → (N, w, 16)
        x_data = x_data.reshape((-1, x_data.shape[2]))  # ((N*16), w)

        train_data.append(x_data)

    train_data = np.vstack(train_data)  # Stack sampled data
    _, Vpca = pca_reduction(train_data, tol=5e-6, comp=PCA_COMPONENTS)

    print(f"Computed PCA matrix (Vpca) with shape: {Vpca.shape}")

    return Vpca

def apply_pca_to_dataset(csv_file, output_dir, Vpca, PCA_COMPONENTS):
    
    os.makedirs(output_dir, exist_ok=True)  

    data_info = pd.read_csv(csv_file)

    for _, row in tqdm(data_info.iterrows(), total=len(data_info), desc="Transforming Data"):
        npz_data = np.load(row["file_path"])
        x_data = npz_data["x_data"]
        y_data = npz_data["y_data"]

        x_data, _ = reduce_matrix(x_data, Vpca, PCA_COMPONENTS)  

        save_path = os.path.join(output_dir, os.path.basename(row["file_path"]))
        np.savez(save_path, x_data=x_data, y_data=y_data)
        # print(f"Saved PCA-transformed test file: {save_path}")

def subj_list_task(task, df):
    """Creates list of subjects to load based on the task"""

    subset = df[df["Group"].isin(class_groups[task])]

    subject_list = tuple(zip(subset['participant_id'], subset['Group']))

    train, test = train_test_split(subject_list, test_size = 0.1, random_state=42, stratify=subset["Group"])

    print(f"Task: {task}")
    print(f"Number of Subjects in Train set {len(train)}")
    print(f"Number of Subjects in Train set {len(test)}")

    return train, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse input arguments for the script.")

    parser.add_argument("--window", type=int, default=1000, help="Window size (default: 1000)")
    parser.add_argument("--percentage_overlap", type=int, default=4, help="Percentage of overlap (default: 0.25)")
    parser.add_argument("--task", type=str, default="A_vs_C", help="Task name (default: 'A_vs_C')")
    parser.add_argument("--pca_components", type=int, default=50, help="Number of PCA components (default: 50)")

    args = parser.parse_args()

    DATASET_DIR = "/home/marta/Documenti/milt_np_dataset"
    OUTPUT_DATA_DIR = "/home/marta/Documenti/data-milt-preprocessed/"
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    WINDOW = args.window
    PERCENTAGE_OVERLAP = args.percentage_overlap
    OVERLAP = WINDOW // PERCENTAGE_OVERLAP
    TASK = args.task
    # OVERLAP = 0
    PCA_COMPONENTS = args.pca_components
    # Assign parsed values to variables

    print(f"WINDOW: {WINDOW}")
    print(f"PERCENTAGE_OVERLAP: {PERCENTAGE_OVERLAP}")
    print(f"TASK: {TASK}")
    print(f"PCA_COMPONENTS: {PCA_COMPONENTS}")

    ## CLASSES
    # A	"Alzheimer Disease Group"
    # F	"Frontotemporal Dementia Group"
    # C	"Healthy Group"

    # Loading data and computing crops
    df = pd.read_csv("/home/marta/Documenti/milt_dataset/datatset/participants.tsv",sep="\t")

    class_groups = {
        "A_vs_C": ["A", "C"],
        "A_vs_F": ["A", "F"],
        "F_vs_C": ["F", "C"],
        "A_vs_F_vs_C": ["A", "F", "C"]
    }
    # function that creates the list of subjects to load based on the task
    train_subj_list, test_subj_list = subj_list_task(TASK, df)

    test_path = os.path.join(OUTPUT_DATA_DIR, f"test_w{WINDOW}_ovr{OVERLAP}_{TASK}")
    train_path = os.path.join(OUTPUT_DATA_DIR, f"train_w{WINDOW}_ovr{OVERLAP}_{TASK}")
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)


    train_config = f"config/train_w{WINDOW}_ovr{OVERLAP}_{TASK}.csv"
    test_config = f"config/test_w{WINDOW}_ovr{OVERLAP}_{TASK}.csv"

    precompute_crops(train_subj_list, 
                    window=WINDOW, 
                    DATASET_DIR=DATASET_DIR, 
                    save_dir=train_path,
                    csv_file_name= train_config, 
                    task=TASK, overlap=OVERLAP)

    precompute_crops(test_subj_list, 
                    window=WINDOW, 
                    DATASET_DIR=DATASET_DIR, 
                    save_dir=test_path, 
                    csv_file_name= test_config,
                    task=TASK, overlap=OVERLAP)
    
    test_pca_path = os.path.join(OUTPUT_DATA_DIR, f"test_w{WINDOW}_ovr{OVERLAP}_pca{PCA_COMPONENTS}_{TASK}")
    train_pca_path = os.path.join(OUTPUT_DATA_DIR, f"train_w{WINDOW}_ovr{OVERLAP}_pca{PCA_COMPONENTS}_{TASK}")
    os.makedirs(test_pca_path, exist_ok=True)
    os.makedirs(train_pca_path, exist_ok=True)

    print("Processing Vpca on the Training set")
    Vpca = compute_vpca(train_config, train_pca_path, PCA_COMPONENTS=50)

    print("Processing and saving Training dataset with pca applied")
    apply_pca_to_dataset(train_config, train_pca_path, Vpca, PCA_COMPONENTS=50)
    print("Processing and saving Testing dataset with pca applied")
    apply_pca_to_dataset(test_config, test_pca_path, Vpca, PCA_COMPONENTS=50)
    
