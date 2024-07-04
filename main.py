import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import build_transformer

class SpatiotemporalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        event = self.data.iloc[idx]
        x = event['x']
        y = event['y']
        time = event['time']
        intensity = event['intensity']
        return torch.tensor([[x, y, time]], dtype=torch.float32), torch.tensor([intensity], dtype=torch.float32)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets).squeeze(1)
    return inputs, targets

# Load the dataset
dataset = SpatiotemporalDataset('synthetic_spatiotemporal_events.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Define hyperparameters
src_input_dim = 3 
tgt_input_dim = 1 
src_seq_len = 1
tgt_seq_len = 1
d_model = 512
N = 6
h = 8
dropout = 0.1
d_ff = 2048
epochs = 10
learning_rate = 0.001

model = build_transformer(src_input_dim, tgt_input_dim, src_seq_len, tgt_seq_len, d_model, N, h, dropout, d_ff)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()

        src_mask = None
        tgt_mask = None
        encoder_output = model.encode(inputs, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, targets.unsqueeze(1), tgt_mask)
        predictions = model.project(decoder_output)
        loss = criterion(predictions.squeeze(), targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")

# Testing loop
model.eval()
total_loss = 0.0
with torch.no_grad():
    for inputs, targets in dataloader:
        src_mask = None
        tgt_mask = None
        encoder_output = model.encode(inputs, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, targets.unsqueeze(1), tgt_mask)
        predictions = model.project(decoder_output)
        loss = criterion(predictions.squeeze(), targets)
        total_loss += loss.item()

print(f"Test Loss: {total_loss/len(dataloader)}")



