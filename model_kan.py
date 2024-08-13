import math
import numpy as np
from kan import KAN
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from tqdm.auto import tqdm

    
def hawkes_intensity_at_point(event_times, event_locations, mu, alpha, beta, gamma, point_time, point_location):
    base_intensity = mu
    # Compute temporal excitation
    time_diffs = point_time - event_times
    excitation_temporal = alpha * torch.exp(-beta * time_diffs)
    # print('inside intensity function with time diff = ', time_diffs)
    # Compute spatial excitation
    spatial_diffs = point_location - event_locations
    spatial_distances = torch.norm(spatial_diffs, dim=-1)
    excitation_spatial = torch.exp(-gamma * spatial_distances)
    # print('inside intensity function with spatial_excitation = ', spatial_diffs)

    # Total excitation
    excitation = excitation_temporal * excitation_spatial
    excitation_sum = excitation.sum()
    intensity = base_intensity + excitation_sum
    return intensity




def calculate_upper_bound(event_times, event_locations, mu, alpha, beta, gamma, T, spatial_range):
    max_intensity = 0.0

    for t in torch.arange(0, T, 0.1):  # Discretize time interval
        for x in torch.arange(spatial_range[0], spatial_range[1], 0.1):
            for y in torch.arange(spatial_range[2], spatial_range[3], 0.1):
                point_time = t
                point_location = torch.tensor([x, y])
                intensity = hawkes_intensity_at_point(event_times, event_locations, mu, alpha, beta, gamma, point_time, point_location)
                if intensity > max_intensity:
                    max_intensity = intensity

    return max_intensity


def generate_event_thinning(event_data, T, model, spatial_range):
    batch_size = event_data.shape[0]
    all_predicted_events = []
    all_rejected_events = []

    for i in range(batch_size):
        # Extract event times and locations
        current_time = event_data[i, -1, -1].item()
        current_location = event_data[i, -1, :-1]
        times = event_data[i, :, -1].tolist()
        locations = event_data[i, :, :-1].tolist()

        predicted_events = []
        rejected_events = []

        # Predict Hawkes process parameters using the model
        mu, alpha, beta, gamma = model(event_data[i].unsqueeze(0), None)
        # Flatten the parameters
        mu = mu.squeeze().item()
        alpha = alpha.squeeze().item()
        beta = beta.squeeze().item()
        gamma = gamma.squeeze().item()
        
        # mu =  0.1
        # alpha = 0.3   
        # beta =  0.3    
        # gamma=  1.0

        

        # Calculate upper bound
        upper_bound = calculate_upper_bound(
            event_data[i, :, -1], event_data[i, :, :-1],
            mu, alpha, beta, gamma, T[i].item(),
            [
                event_data[i, :, 0].min(), event_data[i, :, 0].max(),
                event_data[i, :, 1].min(), event_data[i, :, 1].max()
            ]
        )
        
        upper_bound = upper_bound * 1.5
        
        while current_time < T[i].item():
            inter_event_time = torch.distributions.Exponential(upper_bound).sample().item()
            candidate_time = current_time + inter_event_time
            if candidate_time > T[i].item():
                break

            # Generate candidate location using uniform distribution
            candidate_location = torch.distributions.Uniform(
                current_location - spatial_range / 2,
                current_location + spatial_range / 2
            ).sample().squeeze(0)
            # print('the event_data: ', event_data[i, :, -1].shape)
            print('mu: alpha beta gamma: ', mu, alpha, beta, gamma)
            # print('candidate location: ', candidate_location)
            # print('candidate time: ', candidate_time)
            # Calculate the intensity at the candidate point
            candidate_intensity = hawkes_intensity_at_point(
                event_data[i, :, -1], event_data[i, :, :-1],
                mu, alpha, beta, gamma, candidate_time, candidate_location
            )

            acceptance_prob = candidate_intensity / upper_bound
            
            if torch.rand(1).item() < acceptance_prob:
                predicted_events.append(torch.cat([candidate_location, torch.tensor([candidate_time]), torch.tensor([candidate_intensity])]))
                current_time = candidate_time
                current_location = candidate_location
            else:
                rejected_events.append(torch.cat([candidate_location, torch.tensor([candidate_time]), torch.tensor([candidate_intensity])]))

        if len(predicted_events) == 0:
            candidate_time = current_time + torch.distributions.Exponential(upper_bound).sample().item()
            candidate_location = torch.distributions.Uniform(
                current_location - spatial_range / 2,
                current_location + spatial_range / 2
            ).sample().squeeze(0)
            candidate_intensity = hawkes_intensity_at_point(
                event_data[i, :, -1], event_data[i, :, :-1],
                mu, alpha, beta, gamma, candidate_time, candidate_location
            )
            predicted_events.append(torch.cat([candidate_location, torch.tensor([candidate_time]), torch.tensor([candidate_intensity])]))
            current_time = candidate_time
            current_location = candidate_location
            times.append(candidate_time)
            locations.append(candidate_location.tolist())

        all_predicted_events.append(torch.stack(predicted_events))
        if rejected_events:
            all_rejected_events.append(torch.stack(rejected_events))

    return all_predicted_events, all_rejected_events



class InputEmbedding(nn.Module):
    def __init__(self, d_model, input_dim):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
   
    def forward(self, x):
        # (batch_size, seq_len, input_dim) => (batch_size, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

    

class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p = dropout)
        
        pe = torch.zeros(seq_len, d_model)
        
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # (batch_size, seq_len, d_model) => (batch_size, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
        
            

class FeedForward(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)      
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model 
        self.h = h 
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) 
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False) 
        self.w_o = nn.Linear(d_model, d_model, bias=False) 
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
   
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    
     
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(d_model, h, dropout)
        self.norm_1 = LayerNormalization()
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm_2 = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src2 = self.norm_1(src)
        src = src + self.dropout(self.attention(src2, src2, src2, src_mask))
        src2 = self.norm_2(src)
        src = src + self.dropout(self.ff(src2))
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, N: int, h: int, d_ff: int, dropout: float, seq_len: int):
        super().__init__()
        self.embedding = InputEmbedding(d_model, input_dim)
        self.pe = PositionEncoding(d_model, seq_len, dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        self.norm = LayerNormalization()

    def forward(self, src, src_mask):
        src = self.embedding(src)
        src = self.pe(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)


class KANDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kan_layers: int):
        super(KANDecoder, self).__init__()
        # Define the KAN model
        # width is a list of layer sizes, so we can base it on the input_dim, kan_layers, and output_dim
        # grid, k, and seed are hyperparameters you can tune based on your specific task
        self.kan = KAN(width=[input_dim, kan_layers, output_dim, output_dim], grid=5, k=3, seed=0)
        
    def forward(self, x):
        # KAN expects a 2D tensor: (batch_size, input_dim)
        x_flat = x.mean(dim=1)  # Reduce the sequence dimension
        params = F.softplus(self.kan(x_flat))  # Apply softplus to ensure positivity
        mu, alpha, beta, gamma = torch.chunk(params, 4, dim=-1)
        return mu.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1), gamma.squeeze(-1)


# Replace the decoder in the Transformer model with KANDecoder
class KANTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, N: int, h: int, d_ff: int, dropout: float, seq_len: int, latent_dim: int, kan_layers: int):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, N, h, d_ff, dropout, seq_len)
        self.decoder = KANDecoder(d_model, latent_dim * 4, kan_layers)  # latent_dim * 4 because we have 4 outputs (mu, alpha, beta, gamma)

    def forward(self, src, src_mask):
        enc_output = self.encoder(src, src_mask)
        mu, alpha, beta, gamma = self.decoder(enc_output)
        return mu, alpha, beta, gamma
     

    def log_likelihood(self, event_times, event_locations, mu, alpha, beta, gamma, target_times, target_locations, dt=0.1, dx=0.1, dy=0.1):
        log_intensity_sum = 0
        integrated_intensity_sum = 0
        eps = 1e-9  # Small value to prevent log(0)
        
        for batch_idx in range(target_times.shape[0]):
            batch_event_times = event_times[batch_idx]
            batch_event_locations = event_locations[batch_idx]
            batch_mu = mu[batch_idx]
            batch_alpha = alpha[batch_idx]
            batch_beta = beta[batch_idx]
            batch_gamma = gamma[batch_idx]
            
            # Target events
            batch_target_times = target_times[batch_idx]
            batch_target_locations = target_locations[batch_idx]

            for i in range(len(batch_target_times)):
                point_time = batch_target_times[i]
                point_location = batch_target_locations[i]
                
                # Calculate intensity at the event point
                intensity = hawkes_intensity_at_point(
                    batch_event_times, batch_event_locations,
                    batch_mu, batch_alpha, batch_beta, batch_gamma, 
                    point_time, point_location
                )
                log_intensity_sum += torch.log(intensity + eps)

            # Calculate integrated intensity over the observation window
            T = batch_target_times[-1]
            x_min, x_max = batch_event_locations[:, 0].min(), batch_event_locations[:, 0].max()
            y_min, y_max = batch_event_locations[:, 1].min(), batch_event_locations[:, 1].max()
            
            time_grid = torch.arange(0, T, dt)
            x_grid = torch.arange(x_min, x_max, dx)
            y_grid = torch.arange(y_min, y_max, dy)
            
            for t in time_grid:
                for x in x_grid:
                    for y in y_grid:
                        point_time = t
                        point_location = torch.tensor([x, y], dtype=batch_event_locations.dtype)
                        intensity = hawkes_intensity_at_point(
                            batch_event_times, batch_event_locations,
                            batch_mu, batch_alpha, batch_beta, batch_gamma, 
                            point_time, point_location
                        )
                        integrated_intensity_sum += intensity * dt * dx * dy
        
        log_likelihood_value = log_intensity_sum - integrated_intensity_sum
        return log_likelihood_value / target_times.shape[0]
