##TODO 
## main steps alessandrini 
    # preprocessing dei segnali (Standard Scaler)
    # creazione dei crop a partire dal segnale intero in maniera parametrica (window e overlap)
    # oversample per bilanciare le classi e applica del rumore per differenziare 
    # pca (con le funzioni reduce_matrix e adjust_size)
    # training del modello 
        # 20 epoche con early stopping 
        # loss SparseCategoricalCrossentropy(from_logits = True)
        # optimzier adam
        # rmsprop
    # testing del modello 
    
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, Dataloader
from rnn_model import LSTMModel
import numpy as np 
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.preprocessing import StandardScaler
import r_pca
import scipy.io 


print(torch.cuda.is_available())

def pca_reduction(A, tol, comp = 0):
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

def calculate_accuracy(y_pred, y_true):
  
  correct = (y_pred == y_true).sum().item()
  
  return correct / y_true.size(0)

def train(args, model, device, train_loader, optimizer, epoch):
  
  """
  Define Training Step
  """
  
  model.train()
  
  train_loss = 0.0
  pred_list = []
  gt_list = []
  
  
  for batch_idx, (data, target) in enumerate(train_loader):
    
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    output = model(data)
    
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    
    _, y_pred = torch.max(output,1)
    
    pred_list.append(y_pred)
    gt_list.append(target)
    
  pred_list = torch.cat(pred_list)
  gt_list = torch.cat(gt_list)
  
  train_acc = calculate_accuracy(pred_list, gt_list) 
  
  return train_loss / len(train_loader), train_acc, pred_list, gt_list   
      

def validation(model, device, test_loader):
  
  """
  Define Validation Step
  """
    
  model.eval()
  
  val_loss = 0 
  correct = 0 
  
  pred_list = []
  gt_list = []
  
  with torch.no_grad():
    
    for data, target in test_loader:
        
      data, target = data.to(device), target.to(device)
      output = model(data)
      val_loss += F.nll_loss(output, target, reduction='sum').item()
      pred = output.argmax(dim=1, keepdim=True)
      
      pred_list.append(pred)
      gt_list.append(target)
      # correct += pred.eq(target.view_as(pred)).sum().item()
      
  pred_list = torch.cat(pred_list)
  gt_list = torch.cat(gt_list)
           
  val_acc = calculate_accuracy(pred_list, gt_list)
  
  val_loss /= len(test_loader.dataset)
  
  return val_loss, val_acc, pred_list, gt_list 
    
def save_model(model, optimzier, epoch):
  ##TODO salvare sempre l'ultima epoca 
  # salvare la migliore accuracy/o loss minore 
  """
  Function to save model states for epoch 
  """
  
  model_name = model.__class__.name__
  
  model_dir = f"{os.getcwd()}/output/"
  
  if os.path.exists(model_dir):
    pass
  
  else:
    os.mkdir(model_dir)
  
  now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  
  path = os.path.join(model_dir, f"{model_name}_{now}") 
  
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimzier.state_dict()
    # aggiungere il numero di parametri? 
  }, path)
    
    
def main():
  
  writer = SummaryWriter()

  num_epochs = 20 
  device = torch.device("cuda")
  
  train_loader = EegDataset()
  val_loader = EegDataset()
  test_loader = EegDataset()
  
  model = LSTMModel.to(device)
  
  optimizer = optim.adam(model.parameters)
  # scheduler = StepLR(optimizer, step_size=1)
  
  for epoch in range(1, num_epochs):
  
    train_loss, train_acc, train_preds, train_gts = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch)
    val_loss, val_acc, val_preds, val_gts = validation(model, device, val_loader)
    
    save_model(model, optimizer, epoch)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/validation', val_acc, epoch)
    ##TODO calculate also the confusion matrix calcolare solo alla fine quella di sklearn  
        
        
if __name__ == '__main__':
  
  rpca = False
  rpca_mu = 0
  multiscale_pca = False 
  
  main()