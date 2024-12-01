import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset
import numpy as np 
from sklearn.preprocessing import StandardScaler
import r_pca 
import scipy.io
import os 



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

def reduce_matrix(A, V):
  # (N, w, 16) → (N, 16, w) → ((N*16), w) → compute V
  # (N, 16, w) * V → transpose again last dimensions
  B = np.swapaxes(A, 1, 2)  # (N, 16, w)
  C = B.reshape((-1, B.shape[2]))  # ((N*16), w)
  if V is None:
    L, V = pca_reduction(C, 5e-6, comp = 50)
  B = C @ V  # ((N*16), p)
  B = B.reshape((A.shape[0], A.shape[2], B.shape[1]))  # (N, 16, p)
  return np.swapaxes(B, 1, 2), V  # B = (N, p, 16)

def adjust_size(x, y):
  # when flattening the data matrix on the first dimension, y must be made compatible
  if len(x) == len(y): return y
  factor = len(x) // len(y)
  ynew = np.empty((len(x), 1))
  for i in range(0, len(y)):
    ynew[i * factor : (i + 1) * factor] = y[i]
  return ynew

    # funzione create_dataset di alessandrini che mi crea i crop a cui io tolgo solamente la parte di loading del soggetto
def create_dataset(subject_list, dataset_dir, window, overlap, num_columns=16, num_classes=2):

    x_data = np.empty((0, window, num_columns))
    y_data = np.empty((0, 1))  # Labels
    subj_inputs = []  # Tracks number of windows per subject
    
    print('\n### Creating dataset')
    tot_rows = 0
    
    for subject_id, category_label in subject_list:
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
    
    print(f"Total samples: {tot_rows}")
    print(f"x_data shape: {x_data.shape}")
    print(f"y_data shape: {y_data.shape}")
    print(f"Windows per subject: {subj_inputs}")
    print(f"Class distribution: {[np.sum(y_data == cl) for cl in range(num_classes)]}")
    
    return x_data, y_data, subj_inputs

class EegDataset(Dataset):
    def __init__(self, subject_list: list, data_dir: str, data_mode: str, window: int, overlap: int):

        super().__init__()
        
        self.data_mode = data_mode
        self.subject_list = subject_list
        self.data_dir = data_dir
        self.window = window
        self.overlap = overlap
        
        if data_mode not in ["aless", "caueeg", "milt"]:
            raise ValueError("Set one of the following modalities: 'aless', 'caueeg', 'milt'")
    
    def __len__(self):
        return len(self.subject_list)
    
    def __getitem__(self, index):
      
        subject = self.subject_list[index]
        # Format the filename from the tuple
        subject_id = subject[0]  # e.g., '10'
        label = subject[1]       # e.g., 'N'

        # Create the filename (e.g., 'S01_N.npz')
        filename = f"S{subject_id}_{label}.npz"
        file_path = os.path.join(self.data_dir, filename)

        if self.data_mode == "aless":
          
            x_data, y_data, subj_inputs = self._load_aless(file_path)
            
            x_data_reduced, Vpca = reduce_matrix(x_data, None)
            y_data = adjust_size(x_data_reduced, y_data)

            return torch.tensor(x_data_reduced, dtype=torch.float32), \
                   torch.tensor(y_data, dtype=torch.float32)
                  #  , \ subj_inputs
        
        else:
            raise NotImplementedError(f"mode '{self.data_mode}' not implemented.")
    
    def _load_aless(self, file_path):
      
        x_data, y_data, subj_inputs = create_dataset(self.subject_list, self.data_dir, self.window, self.overlap)
        return x_data, y_data, subj_inputs
