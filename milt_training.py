import os

path = "/home/marta/Documenti/eeg-ml-thesis/"
os.chdir(path)

import torch 
torch.set_num_threads(4) 

import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
import pandas as pd 
import random
from sklearn.metrics import classification_report
import argparse 

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_prob=0.5, use_dense1=False):
        super(LSTMModel, self).__init__()
        
        self.use_dense1 = use_dense1
        if use_dense1:
            self.dense1 = nn.Linear(input_dim, hidden_dim)
        
        self.lstm1 = nn.LSTM(hidden_dim if use_dense1 else input_dim, hidden_dim, num_layers=num_layers, 
                             batch_first=True, dropout=dropout_prob if num_layers > 1 else 0, 
                             bidirectional=False)

        self.dropout1 = nn.Dropout(dropout_prob) 

        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, 
                             dropout=dropout_prob if num_layers > 1 else 0) 

        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if self.use_dense1:
            x = self.dense1(x)
        
        # First LSTM layer
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        # Second LSTM layer (keeps last output only)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Keep only last timestep
        
        # Fully connected output
        out = self.fc(out)
        
        return out  # No softmax, since PyTorch's CrossEntropyLoss applies it

def calculate_accuracy(y_pred, y_true):
  
  correct = (y_pred == y_true).sum().item()
  
  return correct / y_true.size(0)

def train(model, device, train_loader, optimizer, epoch):
  print("starting training loop")
  
  """
  Define Training Step
  """
  
  model.train()
  
  train_loss = 0.0
  pred_list = []
  gt_list = []
  
  
  for batch_idx, (data, target) in enumerate(tqdm(train_loader)):

    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    output = model(data)
    

    target = target.squeeze().long()

    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
    
    _, y_pred = torch.max(output,1)
    
    pred_list.append(y_pred)
    gt_list.append(target)
    
  pred_list = torch.cat(pred_list)
  gt_list = torch.cat(gt_list)
  
  train_acc = calculate_accuracy(pred_list, gt_list) 
  print("ended training step")
  return train_loss / len(train_loader), train_acc, pred_list, gt_list 

def validation(model, device, val_loader):
  print("starting validation step")
  
  """
  Define Validation Step
  """
    
  model.eval()
  
  val_loss = 0   
  pred_list = []
  gt_list = []

  criterion = nn.CrossEntropyLoss()
  
  with torch.no_grad():
    
    for data, target in val_loader:
        
      data, target = data.to(device), target.to(device).squeeze().long()
      output = model(data)
      loss = criterion(output, target)
      val_loss += loss.item()
      _, y_pred = torch.max(output,1)
      
      pred_list.append(y_pred)
      gt_list.append(target)
      # correct += pred.eq(target.view_as(pred)).sum().item()
      
  pred_list = torch.cat(pred_list)
  gt_list = torch.cat(gt_list)
           
  val_acc = calculate_accuracy(pred_list, gt_list)
  
  
  return val_loss / len(val_loader.dataset), val_acc, pred_list, gt_list 
 
def test_and_save_confusion_matrix(model, device, loader, file_name):
    model.eval()
    gt_list = []
    pred_list = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device).to(torch.float32), target.to(device).squeeze().long()
            output = model(data).float()
            _, y_pred = torch.max(output, 1)  
            
            pred_list.append(y_pred)
            gt_list.append(target)
            
        pred_list = torch.cat(pred_list)
        gt_list = torch.cat(gt_list)
      
    test_acc = calculate_accuracy(pred_list, gt_list)
    print(f"Test Accuracy: {test_acc:.4f}")   
    gt_list, pred_list = gt_list.cpu().numpy(), pred_list.cpu().numpy()
    # Compute confusion matrix
    cm = confusion_matrix(gt_list, pred_list)
    report_dict = classification_report(gt_list, pred_list, output_dict=True)
    print(report_dict)
    report_df = pd.DataFrame(report_dict).transpose()

    report_df.to_csv(f"output-milt/report_{file_name}", index=True)
    num_classes = cm.shape[0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(num_classes))

    # Plot and save confusion matrix
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(f'output-milt/cm_{file_name}.png')
    plt.show()   

def seed_everything(seed: int) -> None:
    """
    Set all seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True      

def save_best_model(model, optimizer, epoch, path):
    """
    Saves the best model based on validation loss.
    Overwrites the existing file if the new model is better.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


class PrecomputedPCAEeGDataset(Dataset):
    def __init__(self, csv_file, precomputed_dir, split="train"):
        self.data_info = pd.read_csv(csv_file)
        self.data_info = self.data_info[self.data_info["split"] == split].reset_index(drop=True)
        self.precomputed_dir = precomputed_dir

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        file_name = os.path.basename(row["file_path"])
        file_path = os.path.join(self.precomputed_dir, file_name)  

        npz_data = np.load(file_path)
        x_data = npz_data["x_data"]  
        y_data = npz_data["y_data"]  

        x_data = torch.tensor(x_data, dtype=torch.float32).squeeze(0) 

        y_data = torch.tensor(y_data, dtype=torch.long).squeeze() 

        return x_data, y_data
    
def plot_training_history(history, file_name):

    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r*-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r*-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output-milt/history_{file_name}")
    plt.show()


if __name__ == '__main__':
    print("Training Started")
    parser = argparse.ArgumentParser(description="Parse input arguments for the script.")

    parser.add_argument("--window", type=int, default=1000, help="Window size (default: 1000)")
    parser.add_argument("--percentage_overlap", type=int, default=4, help="Percentage of overlap (default: 0.25)")
    parser.add_argument("--task", type=str, default="A_vs_C", help="Task name (default: 'A_vs_C')")
    parser.add_argument("--pca_components", type=int, default=50, help="Number of PCA components (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Set seed for reproducibilty")
    args = parser.parse_args()

    base_path = "/home/marta/Documenti/data-milt-preprocessed/"

    WINDOW = args.window
    PERCENTAGE_OVERLAP = args.percentage_overlap
    OVERLAP = WINDOW // PERCENTAGE_OVERLAP
    TASK = args.task
    SEED = args.seed
    # OVERLAP = 0
    PCA_COMPONENTS = 50
    num_epochs = 20
    print(f"window: {WINDOW}")
    print(f"overlap: {OVERLAP}")
    print(f"number pca components: {PCA_COMPONENTS}")

    seed_everything(SEED)

    test_pca_path = os.path.join(base_path, f"test_w{WINDOW}_ovr{OVERLAP}_pca{PCA_COMPONENTS}_{TASK}")
    train_pca_path = os.path.join(base_path, f"train_w{WINDOW}_ovr{OVERLAP}_pca{PCA_COMPONENTS}_{TASK}")

    train_config = f"config/train_w{WINDOW}_ovr{OVERLAP}_{TASK}.csv"
    test_config = f"config/test_w{WINDOW}_ovr{OVERLAP}_{TASK}.csv"

    train_dataset = PrecomputedPCAEeGDataset(csv_file=train_config, precomputed_dir=train_pca_path, split="train")
    val_dataset = PrecomputedPCAEeGDataset(csv_file=train_config, precomputed_dir=train_pca_path, split="val")
    test_dataset = PrecomputedPCAEeGDataset(csv_file=test_config, precomputed_dir=test_pca_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,  num_workers=4)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,  num_workers=4)


    for batch_x, batch_y in train_loader:
        print(batch_x.shape, batch_y.shape)
        break 

    # call model and training
    input_dim = 19        
    hidden_dim = 8        
    output_dim = 2    
    window_size = 20      
    dropout_prob = 0.5 
    device = torch.device("cuda")
    model = LSTMModel(input_dim, hidden_dim, output_dim, dropout_prob=dropout_prob, use_dense1=False)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    best_val_acc = float('inf')  

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    file_name = f"w{WINDOW}_ovr{OVERLAP}_pca{PCA_COMPONENTS}_seed{SEED}_{TASK}"
    model_name = file_name + ".pth"
    best_model_path = os.path.join(os.getcwd(), "output-milt", model_name)

    for epoch in range(1, num_epochs + 1):

        print(f"\nProcessing epoch number: {epoch}")
        train_loss, train_acc, train_preds, train_gts = train(model, device, train_loader, optimizer, epoch)

        print(f"Training Accuracy: {train_acc:.4f} - Loss: {train_loss:.4f}")
        val_loss, val_acc, val_preds, val_gts = validation(model, device, val_loader)

        print(f"Validation Accuracy: {val_acc:.4f} - Loss: {val_loss:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_model(model, optimizer, epoch, best_model_path)
            print(f"Best model updated at epoch {epoch} with loss {best_val_acc:.4f}")


    # Save training history
    history_name = file_name + ".npy"
    history_file = os.path.join(os.getcwd(), "output-milt", history_name)
    plot_training_history(history, file_name)
    np.save(history_file, history)

    print(f"\nTraining history saved at {history_file}")
    cm_name = file_name + ".png"
    test_and_save_confusion_matrix(model, device, test_loader, file_name )