import pandas as pd
import torch
from model import build_transformer


src_input_dim = 3 
tgt_input_dim = 3
src_seq_len = 10
tgt_seq_len = 1
d_model = 512
N = 6
h = 8
dropout = 0.1
d_ff = 2048

model = build_transformer(src_input_dim, tgt_input_dim, src_seq_len, tgt_seq_len, d_model, N, h, dropout, d_ff)
model.load_state_dict(torch.load('best_transformer_model.pth'))
model.eval()
    
# Example sequence of events (adjust as per actual data format)
example_events = pd.DataFrame({
    'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'y': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
    'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

# Convert to tensor
example_tensor = torch.tensor(example_events.values, dtype=torch.float32).unsqueeze(0)

# Forward pass through the model
with torch.no_grad():
    src_mask = None
    tgt_mask = None
    predictions, z = model(example_tensor, example_tensor, src_mask, tgt_mask)

print("Predictions:", predictions)
