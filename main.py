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
 
print(torch.cuda.is_available())

class RNN(nn.Module):
    ## struttura di alessandrini 
    # lstm (None, window_size, 8)
    # drop (None, window_size, 8)
    # lstm (None, 8)
    # drop (None, 8)
    # dense (None, num_classes)
    
    def __init__(self):
        pass
    
    def forward(self, x):
        pass

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
            return self._load_caueeg(index)
        
        if self.data_mode == "aless":
            return self._load_aless(index)
        
        if self.data_mode == "milt":
            return self._load_milt(index)
        
    
    def _load_caueeg(self, index):
        # qui leggerei il dataset e ritornerei i dati X e y 
        pass
    
    def _load_aless(self, index):
        # qui leggerei il dataset e ritornerei i dati X e y 
        
        pass
    
    def _load_milt(self, index):
        # qui leggerei il dataset e ritornerei i dati X e y 
        
        pass

def train():
    pass

def test():
    pass

def main():
    pass

if __name__ == '__main__':
    main()