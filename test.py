import torch
import torch.nn.functional as F
import numpy as np
from model import Transformer, generate_event_thinning

# Load the trained model
model_path = 'best_transformer_model.pth'
src_input_dim = 3  # Now includes location (2) + time (1)
src_seq_len = 10
d_model = 512
N = 6
h = 8
dropout = 0.1
d_ff = 2048
latent_dim = 1
# Instantiate and load the model
model = Transformer(src_input_dim, d_model, N, h, d_ff, dropout, src_seq_len, latent_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Example usage with dummy data
# Dummy event times and locations for testing (Batch size, Sequence length, Features)
dummy_event_times = torch.tensor([[1.2, 2.5, 2.8, 3.7, 4.2, 5.0, 5.5, 5.9, 6.3, 7.2] for _ in range(5)], dtype=torch.float32)  # (Batch_size, Seq_len)
dummy_event_locations = torch.tensor([[[i, i + 1] for i in range(10)] for _ in range(5)], dtype=torch.float32)  # (Batch_size, Seq_len, 2)

# Combine times and locations into a single tensor (Batch_size, Seq_len, 3)
dummy_event_data = torch.cat([dummy_event_locations, dummy_event_times.unsqueeze(-1)], dim=-1)  # (Batch_size, Seq_len, 3)

# Dummy T values (final time in the sequence for each batch)
dummy_T = torch.tensor([7.3, 7.8, 8.1,8.8, 9.4], dtype=torch.float32)  # (Batch_size,)

# Define spatial range (as the model was trained on latitude and longitude, using a dummy range)
spatial_range = torch.tensor([1.0, 5.0])

# Predict future events using the thinning method
all_predicted_times_tensor, all_predicted_locations_tensor = generate_event_thinning(
    dummy_event_data,
    dummy_T,
    model,
    spatial_range
)

# Print predicted times and locations
print("Predicted Event Times:")
print(all_predicted_times_tensor)

print("Rejected Event Locations:")
print(all_predicted_locations_tensor)
