import torch
from torch.utils.data import Dataset
import pandas as pd

class NYPDSpatiotemporalDataset(Dataset):
    def __init__(self, csv_file, seq_len):
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len + 1
        self.data = self.data.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATE', 'TIME'])
        


        # Combine date and time into a single datetime column and convert to timestamp
        self.data['DATETIME'] = pd.to_datetime(self.data['DATE'] + ' ' + self.data['TIME'])
        self.data['TIMESTAMP'] = self.data['DATETIME'].apply(lambda x: x.timestamp())

        self.data = self.data.sort_values(by='TIMESTAMP')
        # Normalize the timestamps using min-max normalization
        min_timestamp = self.data['TIMESTAMP'].min()
        max_timestamp = self.data['TIMESTAMP'].max()
        self.data['TIMESTAMP'] = (self.data['TIMESTAMP'] - min_timestamp) / (max_timestamp - min_timestamp)

        lat_min, lat_max = self.data['LATITUDE'].min(), self.data['LATITUDE'].max()
        lon_min, lon_max = self.data['LONGITUDE'].min(), self.data['LONGITUDE'].max()
        self.data['LATITUDE'] = (self.data['LATITUDE'] - lat_min) / (lat_max - lat_min)
        self.data['LONGITUDE'] = (self.data['LONGITUDE'] - lon_min) / (lon_max - lon_min)


        # Select relevant columns
        self.data = self.data[['LATITUDE', 'LONGITUDE', 'TIMESTAMP']]


    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        events = self.data.iloc[idx:idx + self.seq_len]
        x = torch.tensor(events[['LATITUDE', 'LONGITUDE', 'TIMESTAMP']].values[:self.seq_len - 1], dtype=torch.float32)
        y = torch.tensor(events[['LATITUDE', 'LONGITUDE', 'TIMESTAMP']].values[-1], dtype=torch.float32)
        return x, y
    
class JPEQSpatiotemporalDataset(Dataset):
    def __init__(self, csv_file, seq_len):
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len + 1
        self.data = self.data.dropna(subset=['latitude', 'longitude', 'time'])
        


        # Combine date and time into a single datetime column and convert to timestamp
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data['timestamp'] = self.data['time'].apply(lambda x: x.timestamp())

        self.data = self.data.sort_values(by='timestamp')
        # Normalize the timestamps using min-max normalization
        min_timestamp = self.data['timestamp'].min()
        max_timestamp = self.data['timestamp'].max()
        self.data['timestamp'] = (self.data['timestamp'] - min_timestamp) / (max_timestamp - min_timestamp)

        lat_min, lat_max = self.data['latitude'].min(), self.data['latitude'].max()
        lon_min, lon_max = self.data['longitude'].min(), self.data['longitude'].max()
        self.data['latitude'] = (self.data['latitude'] - lat_min) / (lat_max - lat_min)
        self.data['longitude'] = (self.data['longitude'] - lon_min) / (lon_max - lon_min)


        # Select relevant columns
        self.data = self.data[['latitude', 'longitude', 'timestamp']]


    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        events = self.data.iloc[idx:idx + self.seq_len]
        x = torch.tensor(events[['latitude', 'longitude', 'timestamp']].values[:self.seq_len - 1], dtype=torch.float32)
        y = torch.tensor(events[['latitude', 'longitude', 'timestamp']].values[-1], dtype=torch.float32)
        return x, y




def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets

