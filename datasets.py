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
  # print('PCA truncation', dmin, '->', p)
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

class EegDataset(Dataset):
    
    def __init__(self, 
                 file_paths, 
                #  labels, 
                 create_dataset_crop, 
                 window, 
                 overlap):
        
        super().__init__()
        self.file_paths = file_paths
        # self.labels = labels
        self.create_dataset_crop = create_dataset_crop
        self.window = window
        self.overlap = overlap
        
        self.crops_index = self._compute_crops_index()
    
    def _compute_crops_index(self):
        crops_index = []
        for file_idx, (file_path) in enumerate(self.file_paths):
            # print(f"file_path: {file_path}")
            crops, _, _ = self.create_dataset_crop(file_path, self.window, self.overlap)
            
            num_crops = len(crops)
            
            crops_index.extend([(file_idx, crop_idx) for crop_idx in range(num_crops)])
            
        return crops_index
    
    def __len__(self):
        return len(self.crops_index)
    
    def __getitem__(self, idx):
        
        file_idx, crop_idx = self.crops_index[idx]
        file_path = self.file_paths[file_idx]
        
        crops, labels, _ = self.create_dataset_crop(file_path, self.window, self.overlap)
        x_data_reduced, Vpca = reduce_matrix(crops, None)
        labels = adjust_size(x_data_reduced, labels)
        # print(np.unique(label[0]))
        # print(label.shape)
        crop = x_data_reduced[crop_idx]
        label = labels[0] 
        
        label = torch.tensor(label).long().squeeze().unsqueeze(0)        
        # label = self.labels[file_idx]
        
        return torch.tensor(crop).float(), label
