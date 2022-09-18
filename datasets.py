import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
                    
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
            
        self.mode = mode
        self.dataset_size = len(data) 
        
        if not target_only:
            feats = list(range(len(data.shape[1])))
            
        if mode == 'test':
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, feats]
            
        if mode == 'train':
            indices = [i for i in range(len(data)) if i % 10 != 0]
        elif mode == 'dev':
            indices = [i for i in range(len(data)) if i % 10 == 0]
            
        self.data = torch.FloatTensor(data[indices])
        self.target = torch.FloatTensor(target[indices])
        
        self.dim = self.data.shape[1]
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)



    
    



