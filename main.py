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
# from torch.utils.data import Dataset, Dataloader
from rnn_model import LSTMModel
from torch.utils.data import DataLoader
import numpy as np 
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import r_pca
import scipy.io 
from datasets import EegDataset
import matplotlib.pyplot as plt
import random


print(torch.cuda.is_available())

def calculate_accuracy(y_pred, y_true):
  
  correct = (y_pred == y_true).sum().item()
  
  return correct / y_true.size(0)

def train(model, device, train_loader, optimizer, epoch):
  
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
    
    loss = nn.CrossEntropyLoss(output, target)
    train_loss += loss.item()
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
      loss = nn.CrossEntropyLoss(output, target)
      val_loss += loss.item()
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

def test_and_save_confusion_matrix(model, device, loader):
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    
    # Plot and save confusion matrix
    disp.plot(cmap='viridis')
    plt.title('Confusion Matrix')
    plt.savefig('output/confusion_matrix.png')
    plt.show()   
    
def main():
  
  writer = SummaryWriter()

  num_epochs = 20 
  batch_size = 1
  device = torch.device("cuda")
  
  ## CREATE DATASET FOR ALESSANDRINI 
  ##TODO refactor the code
  subj_list = (
      tuple((f'{i:02d}', 'N') for i in range(1, 16)) +  # Normal subjects, S01 to S15
      tuple((f'{i:02d}', 'AD') for i in range(1, 21))   # Alzheimer's subjects, S01 to S20
  )

  subjs_test = (0, 1, 15, 16, 17)  
  test_subject_list = [subj_list[i] for i in subjs_test]

  train_val_subjects = [subj for i, subj in enumerate(subj_list) if i not in subjs_test]

  normal_subjects = [subj for subj in train_val_subjects if subj[1] == 'N']
  ad_subjects = [subj for subj in train_val_subjects if subj[1] == 'AD']

  random.seed(42)  
  random.shuffle(normal_subjects)
  random.shuffle(ad_subjects)

  split_index_normal = int(0.8 * len(normal_subjects))
  split_index_ad = int(0.8 * len(ad_subjects))

  train_normal = normal_subjects[:split_index_normal]
  val_normal = normal_subjects[split_index_normal:]

  train_ad = ad_subjects[:split_index_ad]
  val_ad = ad_subjects[split_index_ad:]

  train_subject_list = train_normal + train_ad
  val_subject_list = val_normal + val_ad

  random.shuffle(train_subject_list)
  random.shuffle(val_subject_list)
  
  ## DATASET PARAMETERS 
  dataset_dir = '/home/marta/Documenti/eeg_rnn_repo/rnn-eeg-ad/eeg2'
  data_mode = 'aless'
  window = 256
  overlap = 25
  
  train_dataset = EegDataset(train_subject_list, dataset_dir, data_mode, window, overlap)
  val_dataset = EegDataset(val_subject_list, dataset_dir, data_mode, window, overlap)
  test_dataset = EegDataset(test_subject_list, dataset_dir, data_mode, window, overlap)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
  
  
  ## MODEL CONFIGURATIONS
  input_dim = 16        
  hidden_dim = 8        
  output_dim = 2    
  window_size = 20      
  dropout_prob = 0.5    
  model = LSTMModel(input_dim, hidden_dim, output_dim, window_size, dropout_prob=dropout_prob)
  
  model = model.to(device)
  
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  # scheduler = StepLR(optimizer, step_size=1)
  
  for epoch in range(1, num_epochs):
  
    train_loss, train_acc, train_preds, train_gts = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch)
    val_loss, val_acc, val_preds, val_gts = validation(model, device, val_loader)
    
    save_model(model, optimizer, epoch)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
  test_and_save_confusion_matrix(model, device, test_loader) 
       
if __name__ == '__main__':
  
  main()