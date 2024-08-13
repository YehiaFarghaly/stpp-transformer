import torch
import torch.nn.functional as F
import numpy as np
from model import Transformer, generate_event_thinning, hawkes_intensity_at_point
from tqdm.auto import tqdm
from plot import plot_lambst_static
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
kan_layers = 4 

# Instantiate and load the model
model = Transformer(src_input_dim, d_model, N, h, d_ff, dropout, src_seq_len, latent_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Example usage with dummy data
# Dummy event times and locations for testing (Batch size, Sequence length, Features)
dummy_event_times = torch.tensor([[1.2, 2.1, 3.8, 2.5, 3.6, 4.7, 5.5, 6.9, 6.3, 5.2]], dtype=torch.float32)  # (Batch_size, Seq_len)
dummy_event_locations = torch.tensor([[[2.67, 1.54], [5.67, 12.54], [1.02, 4.54], [12.527, 10.524], [19.7, 8.824], [2.67, 15.54], [9.67, 1.54], [11.02, 18.54], [6.127, 4.524], [9.7, 18.824] ]], dtype=torch.float32)  # (Batch_size, Seq_len, 2)
# Combine times and locations into a single tensor (Batch_size, Seq_len, 3)
dummy_event_data = torch.cat([dummy_event_locations, dummy_event_times.unsqueeze(-1)], dim=-1)  # (Batch_size, Seq_len, 3)

# Dummy T values (final time in the sequence for each batch)
dummy_T = torch.tensor([15], dtype=torch.float32)  # (Batch_size,)

# Define spatial range (as the model was trained on latitude and longitude, using a dummy range)
spatial_range = [0, 20, 0, 20]

# Predict future events using the thinning method
all_predicted_events, all_rejected_events = generate_event_thinning(
    dummy_event_data,
    dummy_T,
    model,
    spatial_range
)

accepted_events = torch.cat(all_predicted_events).numpy()
rejected_events = torch.cat(all_rejected_events).numpy()

print('accepted; ', accepted_events)
# print('rejected; ', rejected_events)

def calc_grid_intensity(x_num, y_num, t_num, t_start, t_end):
    mu =  0.008582075242884457
    alpha = 4.829000473022461   
    beta =  3.209417991456576e-01   
    gamma=  9.258946418762207
    x_max = 20
    y_max = 20
    x_min = 0
    y_min = 0
       
    x_range = np.linspace(x_min, x_max, x_num)
    y_range = np.linspace(y_min, y_max, y_num)
    t_range = np.linspace(t_start, t_end, t_num)
        
    intensities = []
    for t in tqdm(t_range):
        lamb_st = np.zeros((x_num, y_num))
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                
                lamb_st[i, j] = hawkes_intensity_at_point(dummy_event_times[0], dummy_event_locations[0], mu, alpha, beta, gamma, t, torch.tensor([x, y]))
        intensities.append(lamb_st)
    return intensities, x_range, y_range, t_range
intensities, x_range, y_range, t_range = calc_grid_intensity(21, 21, 31, 6, 15)
# Print predicted times and locations
plot_lambst_static(intensities, accepted_events, rejected_events,
                   x_range, y_range, t_range, 
                   fps=3, fn='result.mp4')
