from pandas import read_csv

from sklearn.preprocessing import LabelEncoder

from torch.utils.data import random_split
from torch.utils.data import Dataset



class CSVDataset(Dataset):
    def __init__(self, csv_path, **kwargs):
        df = read_csv(csv_path, **kwargs)
        self._x = df.values[:,:-1].astype('float32')
        
        raw_labels = df.values[:,-1]
        self._y = LabelEncoder() \
            .fit_transform(raw_labels) \
            .astype('float32') \
            .reshape((-1,1))

    def __len__(self):
        return len(self._x)
    
    def __getitem__(self, idx):
        return [self._x[idx], self._y[idx]]
    
    def get_splits(self, n_test=0.33):
        test_size = round(n_test * len(self))
        train_size = len(self) - test_size
        return random_split(self, [train_size, test_size])