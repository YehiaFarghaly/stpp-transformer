import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from model_kan import KANTransformer, generate_event_thinning, hawkes_intensity_at_point
from model import Transformer, generate_event_thinning, hawkes_intensity_at_point
from plot import plot_lambst_static
from data import NYPDSpatiotemporalDataset, collate_fn
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

test_dataset = NYPDSpatiotemporalDataset('test.csv', seq_len=src_seq_len)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)



# Define spatial range (as the model was trained on latitude and longitude, using a dummy range)
all_event_data = []
all_T_values = []

for inputs, targets in test_dataloader:
    all_event_data.append(inputs)
    all_T_values.append(targets[:, -1])  # Last timestamp in the sequence as T

all_event_data = torch.cat(all_event_data)
all_T_values = torch.cat(all_T_values)

spatial_range = [0, 1, 0, 1]

# Predict future events using the thinning method
all_predicted_events, all_rejected_events, mu_values, alpha_values, beta_values, gamma_values = generate_event_thinning(
    all_event_data,
    all_T_values,
    model,
    spatial_range
)


accepted_events = torch.cat(all_predicted_events).numpy()
rejected_events = np.array([])
if len(all_rejected_events) > 0:
    rejected_events = torch.cat(all_rejected_events).numpy()
       


def calc_grid_intensity(x_num, y_num, t_num, t_start, t_end):
       
    x_max = 1
    y_max = 1
    x_min = 0
    y_min = 0
    
    x_range = np.linspace(x_min, x_max, x_num)
    y_range = np.linspace(y_min, y_max, y_num)
    t_range = np.linspace(t_start, t_end, t_num)
    
    all_intensities = []
    batch_size = len(mu_values)
       
    for batch_idx in range(batch_size):
            mu = mu_values[batch_idx]
            alpha = alpha_values[batch_idx]
            beta = beta_values[batch_idx]
            gamma = gamma_values[batch_idx]

            intensities = []
            for t in tqdm(t_range, desc=f"Batch {batch_idx+1}/{batch_size}"):
                lamb_st = np.zeros((x_num, y_num))
                for i, x in enumerate(x_range):
                    for j, y in enumerate(y_range):
                        lamb_st[i, j] = hawkes_intensity_at_point(
                            all_event_data[batch_idx][:, -1],  # event_times
                            all_event_data[batch_idx][:, :-1], 
                            mu, alpha, beta, gamma, t, 
                            torch.tensor([x, y])
                        )
                intensities.append(lamb_st)
            
            all_intensities.append(intensities)
        
    return all_intensities, x_range, y_range, t_range

all_intensities, x_range, y_range, t_range = calc_grid_intensity(15, 15, 25, 2, 70)

plot_lambst_static(all_intensities, accepted_events, rejected_events,
                   x_range, y_range, t_range, 
                   fps=20, fn='result.mp4')
