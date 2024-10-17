##TODO 
## main steps alessandrini 
    # preprocessing dei segnali (Standard Scaler)
    # creazione dei crop a partire dal segnale intero in maniera parametrica (window e overlap)
    # oversample per bilanciare le classi e applica del rumore per differenziare 
    # pca (con le funzioni reduce_matrix e adjust_size)
    # training del modello 
        # 20 epoche con early stopping 
        # loss SparseCategoricalCrossentropy(from_logits = True)
        # rmsprop
    # testing del modello 
    
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, Dataloader
from rnn_model import LSTMModel
import numpy as np 
from sklearn.preprocessing import StandardScaler

print(torch.cuda.is_available())


class EegDataset(Dataset):
    def __init__(self, data_mode: str, offset, split_len):
        super().__init__()
        # data_mode servirebbe per definire il dataset da caricare  
        # altrimenti dovrei standardizzare tutti i file di configurazione 
        if data_mode not in ["aless", "caueeg", "milt"]: 
            raise ValueError("Set one of the following modalities: 'aless', 'caueeg', 'milt'")        
        
        
        # bisognerebbe cambiare questi parametri in base al datset (?)
        # miltiadus avrei i config file in csv mentre per alessandrini il dataset non ha 
        # config files 
        
        self.offset = offset # modalità per caricare i dati caueeg con gli split/offset 
        self.split_len = split_len
        # self.dataset_path = dataset_path
    
    
    def __len__(self):
        return self.split_len
    
    def __getitem__(self, index):
        
        if self.data_mode == "caueeg":
            self._load_caueeg(index)
            crop = self.create_dataset() # crop della registrazione
            eeg_data  = self.pca(crop)  # passo i crop 
            
            return egg_data, label
        
        if self.data_mode == "aless":
            return self._load_aless(index)
        
        if self.data_mode == "milt":
            return self._load_milt(index)
    
    def _load_caueeg(self, index):
        # qui leggerei il dataset e ritornerei i dati X e y 
        pass
    
    def _load_aless(self, index):
        # qui leggerei il dataset e ritornerei i dati X e y 
        # registrazione intera e ritorno il crop 
        # fare la pca sulla registrazione del paziente 
        np.random.seed(42)
        dataset_dir = working_dir + '/eeg2'
        subj_list = tuple((f'{i:02d}', 'N') for i in range(1, 16)) + tuple((f'{i:02d}', 'AD') for i in range(1, 21))
        print(subj_list)
        num_columns = 16
        # return registrazione 
        pass
    
    # funzione create_dataset di alessandrini che mi crea i crop a cui io tolgo solamente la parte di loading del soggetto
    def create_dataset(window, overlap, decimation_factor = 0):
      # Create the input and target data from dataset,
      # according to window and overlap
      # new dataset 4 dec 2021
      # 15 N, 20 AD (resulting indexes: N = 0..14, AD = 15..34)
      #Common signals: ['EEG Fp1', 'EEG Fp2', 'EEG F7', 'EEG F3', 'EEG F4', 'EEG F8', 'EEG T3', 'EEG C3', 'EEG C4', 'EEG T4', 'EEG T5', 'EEG P3', 'EEG P4', 'EEG T6', 'EEG O1', 'EEG O2']

    #   tf.random.set_seed(42)


      x_data = np.empty((0, window, num_columns))
      y_data = np.empty((0, 1))  # labels
      subj_inputs = []  # number of inputs for every subject
      print('\n### creating dataset')
      tot_rows = 0
      for subject in subj_list: # subj_list ora dipende dalla batch size 
        subj_inputs.append(0)
        category = ('N', 'AD').index(subject[1])
        eeg = np.load(f'{dataset_dir}/S{subject[0]}_{subject[1]}.npz')['eeg'].T
        # if spikes: eeg = set_holes(eeg, spikes)
        #scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = StandardScaler()
        eeg = scaler.fit_transform(eeg)
        assert(eeg.shape[1] == num_columns)
        tot_rows += len(eeg)
        # decimation (optional)
        # windowing
        # compute number of windows (lazy way)
        i = 0
        num_w = 0
        while i + window  <= len(eeg):
          i += (window - overlap)
          num_w += 1
        # compute actual windows
        x_data_part = np.empty((num_w, window, num_columns))  # preallocate
        i = 0
        for w in range(0, num_w):
          x_data_part[w] = eeg[i:i + window]
          i += (window - overlap)
          if False: # watermark provenience of every window
            for cc in range(0, num_columns):
              x_data_part[w, 0, cc] = 1000 * (len(subj_inputs) - 1) + cc
        x_data = np.vstack((x_data, x_data_part))
        y_data = np.vstack((y_data, np.full((num_w, 1), category)))
        subj_inputs[-1] += num_w

      print('\ntot samples:', tot_rows)
      print('x_data:', x_data.shape)
      print('y_data:', y_data.shape)
      print('windows per subject:', subj_inputs)
      print('class distribution:', [np.sum(y_data == cl) for cl in range(0, num_classes)])

      return x_data, y_data, subj_inputs
    
    def pca(self):
        # in realtà uso le funzioni della pca di alessandrini 
        pass 
    
    def _load_milt(self, index):
        # qui leggerei il dataset e ritornerei i dati X e y 
        
        pass

def train(args, model, device, train_loader, optimizer, epoch):
    # carico il dataloader 
    
    # cosa prende in pasto il modello? --> prendere un crop alla volta e poi la rnn lo processa in maniera sequenziale
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # devo capire se mettere qui la pca o metterla direttametne nel data loader 
             
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        


def test(model, device, test_loader):
    
    model.eval()
    
    test_loss = 0 
    correct = 0 
    
    with torch.no_grad():
        for data, target in test_loader:
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    
def save_model():
    ##TODO funzione per salvare i parametri del modello durante le epoche 
    
    pass

def calculate_metrics():
    ##TODO definire una funzione per calcolare le metriche durante il training
    # del modello così posso salvare il modello all'epoca migliore 
    
    pass

def main():
    num_epochs = 20 
    device = torch.device("cuda")
    
    train_loader = EegDataset()
    val_loader = EegDataset()
    test_loader = EegDataset()
    
    model = LSTMModel.to(device)
    
    optimizer = optim.adam(model.parameters)
    # scheduler = StepLR(optimizer, step_size=1)
    
    for epoch in range(1, num_epochs):
        train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch)
        test(model, device, val_loader)

if __name__ == '__main__':
    main()