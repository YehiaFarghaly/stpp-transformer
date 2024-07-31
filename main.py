import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from model import Transformer
import pandas as pd
import torch


class NYPDSpatiotemporalDataset(Dataset):
    def __init__(self, csv_file, seq_len):
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len + 1
        self.data = self.data.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATE', 'TIME'])

        # Combine date and time into a single datetime column and convert to timestamp
        self.data['DATETIME'] = pd.to_datetime(self.data['DATE'] + ' ' + self.data['TIME'])
        self.data['TIMESTAMP'] = self.data['DATETIME'].apply(lambda x: x.timestamp())

        # Normalize the timestamps using min-max normalization
        min_timestamp = self.data['TIMESTAMP'].min()
        max_timestamp = self.data['TIMESTAMP'].max()
        self.data['TIMESTAMP'] = (self.data['TIMESTAMP'] - min_timestamp) / (max_timestamp - min_timestamp)

        # Select relevant columns
        self.data = self.data[['LATITUDE', 'LONGITUDE', 'TIMESTAMP']]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        events = self.data.iloc[idx:idx + self.seq_len]
        x = torch.tensor(events[['LATITUDE', 'LONGITUDE', 'TIMESTAMP']].values[:self.seq_len - 1], dtype=torch.float32)
        y = torch.tensor(events[['LATITUDE', 'LONGITUDE', 'TIMESTAMP']].values[-1], dtype=torch.float32)
        return x, y


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets


# Load the dataset
dataset = NYPDSpatiotemporalDataset('database.csv', 10)

# Split the dataset into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Define hyperparameters
src_input_dim = 3 
src_seq_len = 10
d_model = 512
N = 6
h = 8
dropout = 0.1
d_ff = 2048
epochs = 100  # Increase the number of epochs
learning_rate = 0.001
patience = 10  # Early stopping patience
latent_dim = 1

model = Transformer(src_input_dim, d_model, N, h, d_ff, dropout, src_seq_len, latent_dim)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with early stopping
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        src_mask = None
        # Set T to the maximum time in the current batch
        T = targets[:, -1]
        # Forward pass
        mu, alpha, beta, gamma = model(inputs, src_mask)
        # print('results of the prediction mu, alpha, beta, gamma: ', mu, alpha, beta, gamma)
        # Compute log likelihood loss
        event_times = inputs[:, :, -1]  # Assuming time is the last feature in input
        event_locations = inputs[:, :, :-1]  # Assuming the rest are locations
        log_likelihood_loss = -model.log_likelihood(event_times, event_locations, mu, alpha, beta, gamma).mean()
        loss = log_likelihood_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_dataloader)

    # Validate the model
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            src_mask = None
            
            # Set T to the maximum time in the current batch
            T = targets[:, -1]
            
            # Forward pass
            mu, alpha, beta, gamma = model(inputs, src_mask)
            
            # Compute log likelihood loss
            event_times = inputs[:, :, -1]
            event_locations = inputs[:, :, :-1]
            log_likelihood_loss = -model.log_likelihood(event_times, event_locations, mu, alpha, beta, gamma).mean()
            loss = log_likelihood_loss
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_transformer_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

model.load_state_dict(torch.load('best_transformer_model.pth'))
model.eval()
