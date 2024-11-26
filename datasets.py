import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, Dataloader
import numpy as np 



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
      # quanti soggetti carico con il batch 
        return self.split_len
    
    def __getitem__(self, index):
        
        if self.data_mode == "caueeg":

            
            return egg_data, label
        
        if self.data_mode == "aless":
            self._load_aless(index) # la shape sarà la durata della registrazione per il numero di canali
            crop = self.create_dataset() # crop della registrazione
            eeg_data  = self.pca(crop)  # passo i crop 
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
