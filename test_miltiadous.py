# %% [markdown]
# ### IMPORTS

# %%
import os

path = "/home/marta/Documenti/eeg-ml-thesis/"
os.chdir(path)

import torch 
torch.set_num_threads(4) 

import torch.nn as nn 
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

# %%
def precompute_crops(subject_list, window, overlap, DATASET_DIR, num_columns=16, train_dataset=None):
    

    if train_dataset == True:
        save_dir = "/home/marta/Documenti/eeg-ml-thesis/alessandrini-train"
        os.makedirs(save_dir, exist_ok=True)
    elif train_dataset == False:
        save_dir = "/home/marta/Documenti/eeg-ml-thesis/alessandrini-test"
        os.makedirs(save_dir, exist_ok=True)

    for subject_id, category_label in subject_list:
        file_path = f"{DATASET_DIR}/S{subject_id}_{category_label}.npz"
        save_path = f"{save_dir}/S{subject_id}_{category_label}_crops.npz"

        # if os.path.exists(save_path): 
        #    print(f"Skipping {subject_id}, crops already exist.")
        #    continue

        eeg = np.load(file_path)['eeg'].T 

        scaler = StandardScaler()
        eeg = scaler.fit_transform(eeg)

        num_windows = (len(eeg) - window) // (window - overlap) + 1
        x_data = np.empty((num_windows, window, num_columns))

        i = 0
        for w in range(num_windows):
            x_data[w] = eeg[i:i + window]
            i += (window - overlap)

        y_data = np.full((num_windows, 1), (category_label == 'AD')) 

        np.savez(save_path, x_data=x_data, y_data=y_data)
        # print(f"Saved crops for {subject_id} at {save_path}")


def load_npz_data(directory):

    x_list = []
    y_list = []

    for file in os.listdir(directory):
        
        if file.endswith(".npz"):  
            file_path = os.path.join(directory, file)
            data = np.load(file_path)
            
            x_list.append(data['x_data'])  
            y_list.append(data['y_data'])  
    
    x_data = np.vstack(x_list) if x_list else np.array([])
    y_data = np.vstack(y_list) if y_list else np.array([])

    return x_data, y_data

def split_train_val(dataset, test_size=0.2, random_state=42):
    train_indices, val_indices = train_test_split(
        range(len(dataset.crops_index)), 
        test_size=test_size,
        random_state=random_state
    )
    return train_indices, val_indices 

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

def reduce_matrix(A, V, PCA_COMPONENTS):
  # (N, w, 16) → (N, 16, w) → ((N*16), w) → compute V
  # (N, 16, w) * V → transpose again last dimensions
  B = np.swapaxes(A, 1, 2)  # (N, 16, w)
  C = B.reshape((-1, B.shape[2]))  # ((N*16), w)
  if V is None:
    L, V = pca_reduction(C, 5e-6, comp = PCA_COMPONENTS)
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


def oversampling(x_data, y_data, num_classes=2):
  # Duplicate inputs with classes occurring less, so to have a more balanced distribution.
  # It operates on single data windows, so use it on data that have already been split
  #  by subject (typically only on training data).
  x_data_over = x_data.copy()
  y_data_over = y_data.copy()
  occurr = [np.sum(y_data == cl) for cl in range(0, num_classes)]
  for cl in range(0, num_classes):
    if occurr[cl] == max(occurr):
      continue
    mask = y_data[:, 0] == cl
    x_dup = x_data[mask].copy()
    y_dup = y_data[mask].copy()
    while occurr[cl] < max(occurr):
      x_dup_jitter = x_dup + np.random.normal(scale=0.03, size=x_dup.shape)
      how_many = min(len(y_dup), max(occurr) - occurr[cl])
      x_data_over = np.vstack((x_data_over, x_dup_jitter[:how_many]))
      y_data_over = np.vstack((y_data_over, y_dup[:how_many]))
      occurr[cl] += how_many
  return x_data_over, y_data_over



class AlessandriniEegDataset(Dataset):
    def __init__(self, x_data, y_data, PCA_COMPONENTS):
        self.x_data = x_data
        self.y_data = y_data
        self.PCA_COMPONENTS = PCA_COMPONENTS

        # Compute PCA once using training data
        print("Computing PCA matrix for training data...")
        _, self.V_pca = reduce_matrix(self.x_data, None, PCA_COMPONENTS)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]  # Shape: (16, w)
        y = self.y_data[idx]

        # Apply PCA dynamically per sample
        x_pca, _ = reduce_matrix(x[np.newaxis, :, :], self.V_pca, self.PCA_COMPONENTS)  
        x_pca = x_pca.squeeze(0)  # Remove batch dimension

        return torch.tensor(x_pca, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_prob=0.5, use_dense1=False):
        super(LSTMModel, self).__init__()
        
        # Optional Dense Layer Before LSTM (matches TensorFlow's `dense1`)
        self.use_dense1 = use_dense1
        if use_dense1:
            self.dense1 = nn.Linear(input_dim, hidden_dim)
        
        # First LSTM Layer (returns full sequence if second LSTM exists)
        self.lstm1 = nn.LSTM(hidden_dim if use_dense1 else input_dim, hidden_dim, num_layers=num_layers, 
                             batch_first=True, dropout=dropout_prob if num_layers > 1 else 0, 
                             bidirectional=False)

        self.dropout1 = nn.Dropout(dropout_prob) 

        # Second LSTM Layer (if present, returns last output)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, 
                             dropout=dropout_prob if num_layers > 1 else 0) 

        self.dropout2 = nn.Dropout(dropout_prob)

        # Fully Connected Output Layer (No Softmax, since CrossEntropyLoss expects logits)
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
     
 
def test_and_save_confusion_matrix(model, device, loader,cm_name):
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
    print(f"Test Accuracy: {test_acc:.4f}%")    
    # Compute confusion matrix
    cm = confusion_matrix(gt_list.cpu().numpy(), pred_list.cpu().numpy())
    num_classes = cm.shape[0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(num_classes))

    # Plot and save confusion matrix
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(f'output/{cm_name}')
    plt.show()   
         
def save_model(model, optimizer, epoch):
    """
    Function to save model states for a given epoch.
    """
    
    model_name = model.__class__.__name__

    model_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(model_dir, exist_ok=True)  # Creates directory if it doesn't exist

    # Generate filename with timestamp
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(model_dir, f"{model_name}_{now}.pth")  # Add `.pth` for clarity

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

    print(f"Model saved to {path}")

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

    print(f"Best model saved to {path}")

# %%


# %%

def subj_list_task(task, df):

    class_groups = {
        "A_vs_C": ["A", "C"],
        "A_vs_F": ["A", "F"],
        "F_vs_C": ["F", "C"],
        "A_vs_F_vs_C": ["A", "F", "C"]
    }

    subset = df[df["Group"].isin(class_groups[task])]

    subject_list = tuple(zip(subset['participant_id'], subset['Group']))

    train, test = train_test_split(subject_list, test_size = 0.1, random_state=42, stratify=subset["Group"])

    print(f"Task: {task}")
    print(f"Number of Subjects in Train set {len(train)}")
    print(f"Number of Subjects in Train set {len(test)}")

    return train, test

# %% [markdown]
# ### TESTTING

# %%
DATASET_DIR = "/home/marta/Documenti/milt_np_dataset"
WINDOW = 256
OVERLAP = WINDOW // 4
# OVERLAP = 0
PCA_COMPONENTS = 50
num_epochs = 20
print(f"window: {WINDOW}")
print(f"window: {OVERLAP}")
print(f"window: {PCA_COMPONENTS}")

## CLASSES
# A	"Alzheimer Disease Group"
# F	"Frontotemporal Dementia Group"
# C	"Healthy Group"

# Loading data and computing crops



df = pd.read_csv("/home/marta/Documenti/milt_dataset/datatset/participants.tsv",sep="\t")

# function that creates the list of subjects to load based on the task
train_subj_list, test_subj_list = subj_list_task("A_vs_C", df)

# %%

class_groups = {
    "A_vs_C": ["A", "C"],
    "A_vs_F": ["A", "F"],
    "F_vs_C": ["F", "C"],
    "A_vs_F_vs_C": ["A", "F", "C"]
}

def precompute_crops(subject_list, window, overlap, DATASET_DIR, task, train_dataset=None):

    base_dir = "/home/marta/Documenti/eeg-ml-thesis/"
    

    if train_dataset == True:
        save_dir = os.path.join(base_dir,"miltiadous-train")
        os.makedirs(save_dir, exist_ok=True)
    elif train_dataset == False:
        save_dir = os.path.join(base_dir,"miltiadous-test")
        os.makedirs(save_dir, exist_ok=True)
    task_class = class_groups[task]
    label_mapping = {cls: i for i, cls in enumerate(task_class)}

    print(f"Class Mapping:{label_mapping}")
    mapping_file = os.path.join(base_dir, "output-milt",f"class_mapping_{task}.txt")

    with open(mapping_file, "w") as f:
        f.write(f"Task: {task}\n")
        f.write(f"Class mapping: {label_mapping}")


    csv_file_name = f"train_config_file_milt_{task}.csv"
    all_crops = []
  

    for subject_id, category_label in subject_list:
        file_path = f"{DATASET_DIR}/{category_label}/{subject_id}.npy"
        save_path = f"{save_dir}/{subject_id}_{category_label}_crops.npz"
        # if os.path.exists(save_path): 
        #    print(f"Skipping {subject_id}, crops already exist.")
        #    continue
        eeg = np.load(file_path).T 
        scaler = StandardScaler()
        eeg = scaler.fit_transform(eeg)
        num_columns = eeg.shape[1]
        num_windows = (len(eeg) - window) // (window - overlap) + 1
        x_data = np.empty((num_windows, window, num_columns))
        # print(category_label)
        # print(subject_id)
        # print(num_windows)
        i = 0
        for w in range(num_windows):
            x_data[w] = eeg[i:i + window]
            i += (window - overlap)

            all_crops.append((subject_id, w, category_label, save_path))
            # writer.writerow([subject_id, w, category_label])
        y_label = label_mapping[category_label]
        y_data = np.full((num_windows, 1), y_label) 
        np.savez(save_path, x_data=x_data, y_data=y_data)
            # print(f"Saved crops for {subject_id} at {save_path}")
        
    train_ind, val_ind = train_test_split(all_crops, train_size=0.75, random_state=42, shuffle=True)

    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["subject", "crops_number", "label", "split", "file_path"])

        for crop in train_ind:
            writer.writerow([crop[0],crop[1],crop[2],"train", crop[3]])
        for crop in val_ind:
            writer.writerow([crop[0],crop[1],crop[2],"val", crop[3]])


        # writer.writerows(test_csv) 
            
        

# %%
precompute_crops(train_subj_list, window=WINDOW, DATASET_DIR=DATASET_DIR, task="A_vs_C", overlap=OVERLAP, train_dataset=True)
precompute_crops(test_subj_list, window=WINDOW, DATASET_DIR=DATASET_DIR, task="A_vs_C", overlap=OVERLAP, train_dataset=False)

# %%
# Loading crops for oversampling (only training and validation dataset is oversampled)
test_path = "/home/marta/Documenti/eeg-ml-thesis/miltiadous-test"
train_path = "/home/marta/Documenti/eeg-ml-thesis/miltiadous-train"

# %%
class MiltiadousEeGDataset(Dataset):
    def __init__(self, csv_file, split="train", transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.data_info = self.data_info[self.data_info["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        file_path = row["file_path"]
        crop_number = int(row["crops_number"]) - 1  

        npz_data = np.load(file_path)
        x_data = npz_data["x_data"][crop_number]  
        y_data = npz_data["y_data"][crop_number]  

        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.long)  

        if self.transform:
            x_data = self.transform(x_data)

        return x_data, y_data
    

# testare il dataloader senza PCA
csv_file_path = "/home/marta/Documenti/eeg-ml-thesis/train_config_file_milt_A_vs_C.csv"
train_dataset = MiltiadousEeGDataset(csv_file=csv_file_path, split="train")
val_dataset = MiltiadousEeGDataset(csv_file=csv_file_path, split="val")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

for batch_x, batch_y in train_loader:
    print(batch_x.shape, batch_y.shape)
    break 

# %%

print(f"Original dataset size: {X.shape}, Labels distribution: {np.bincount(y.flatten())}")
X_over, y_over = oversampling(X, y)
print(f"Oversampled dataset size: {X_over.shape}, Labels distribution: {np.bincount(y_over.flatten())}")
X_over = torch.tensor(X_over).float()
y_over = torch.tensor(y_over).float()
# Train, val, test split and apply PCA 
X_train, X_val, y_train, y_val = train_test_split(X_over, y_over, train_size = 0.75, random_state=42, shuffle=True)
print(f"training data shape: {X_train.shape}")
print(f"training data shape: {y_train.shape}")
print(f"validation data shape: {X_val.shape}")
print(f"validation data shape: {y_val.shape}")
# X_train, Vpca = reduce_matrix(X_train, None, PCA_COMPONENTS)
# y_train = adjust_size(X_train, y_train)
# X_val, _ = reduce_matrix(X_val, Vpca, PCA_COMPONENTS)
# y_val = adjust_size(X_val, y_val)
# X_test, _ = reduce_matrix(X_test, Vpca.cpu().numpy() if isinstance(Vpca, torch.Tensor) else Vpca, PCA_COMPONENTS)
# y_test = adjust_size(X_test, y_test).astype(np.float32)
# x_data_test = x_data_test.astype(np.float32)
# print(f"training data shape: {X_train.shape}")
# print(f"training data shape: {y_train.shape}")
# print(f"validation data shape: {X_val.shape}")
# print(f"validation data shape: {y_val.shape}")
# Initialize the dataset
train_dataset = AlessandriniEegDataset(X_train, y_train, PCA_COMPONENTS)
val_dataset = AlessandriniEegDataset(X_val, y_val, PCA_COMPONENTS)
test_dataset = AlessandriniEegDataset(X_test, y_test, PCA_COMPONENTS)
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
# Check one batch
for x, y in train_loader:
    print("Batch X shape:", x.shape)  # Expected: (batch_size, PCA_COMPONENTS, 16)
    print("Batch Y shape:", y.shape)  # Expected: (batch_size,)
    break

# %%

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
# scheduler = StepLR(optimizer, step_size=1)
best_val_loss = float('inf')  
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}
file_name = f"{WINDOW}_{OVERLAP}_{PCA_COMPONENTS}"
model_name = file_name + ".pth"
best_model_path = os.path.join(os.getcwd(), "output", model_name)  
for epoch in range(1, num_epochs + 1):
    print(f"\nProcessing epoch number: {epoch}")
    train_loss, train_acc, train_preds, train_gts = train(model, device, train_loader, optimizer, epoch)
    print(f"Training Accuracy: {train_acc:.2f}% - Loss: {train_loss:.4f}")
    val_loss, val_acc, val_preds, val_gts = validation(model, device, val_loader)
    print(f"Validation Accuracy: {val_acc:.2f}% - Loss: {val_loss:.4f}")
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_best_model(model, optimizer, epoch, best_model_path)
        print(f"Best model updated at epoch {epoch} with loss {best_val_loss:.4f}")
# Save training history
history_name = file_name + ".npy"
history_file = os.path.join(os.getcwd(), "output", history_name)
np.save(history_file, history)
print(f"\nTraining history saved at {history_file}")
cm_name = file_name + ".png"
test_and_save_confusion_matrix(model, device, test_loader, cm_name = cm_name)


