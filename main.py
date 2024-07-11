import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import math
from model import build_transformer

class SpatiotemporalDataset(Dataset):
    def __init__(self, csv_file, seq_len):
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        events = self.data.iloc[idx:idx + self.seq_len]
        x = torch.tensor(events[['x', 'y', 'time']].values, dtype=torch.float32)
        y = torch.tensor(events[['x', 'y', 'time']].values[-1], dtype=torch.float32)
        return x, y

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    event_times = inputs[:, :, -1]  
    return inputs, targets, event_times


dataset = SpatiotemporalDataset('synthetic_spatiotemporal_events.csv', 10)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

src_input_dim = 3 
tgt_input_dim = 3
src_seq_len = 10
tgt_seq_len = 1
d_model = 512
N = 6
h = 8
dropout = 0.1
d_ff = 2048
epochs = 100  
learning_rate = 0.001
patience = 10  

model = build_transformer(src_input_dim, tgt_input_dim, src_seq_len, tgt_seq_len, d_model, N, h, dropout, d_ff)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets, event_times in train_dataloader:
        optimizer.zero_grad()
        src_mask = None
        tgt_mask = None
        predictions, z = model(inputs, targets.unsqueeze(1), src_mask, tgt_mask)
        
        regression_loss = criterion(predictions, targets)
        
        log_likelihood_loss = -model.log_likelihood(z, event_times)
        
        loss = regression_loss + log_likelihood_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_dataloader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets, event_times in val_dataloader:
            src_mask = None
            tgt_mask = None
            predictions, z = model(inputs, targets.unsqueeze(1), src_mask, tgt_mask)
            
            regression_loss = criterion(predictions, targets)
            log_likelihood_loss = -model.log_likelihood(z, event_times)
            
            loss = regression_loss + log_likelihood_loss
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
