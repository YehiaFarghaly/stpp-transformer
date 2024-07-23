import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from tqdm.auto import tqdm

    
class LatentProcess(nn.Module):
    def __init__(self, d_model, rnn_hidden_size=256):
        super(LatentProcess, self).__init__()
        self.rnn = nn.GRU(d_model, rnn_hidden_size, batch_first=True)
        self.mu = nn.Linear(rnn_hidden_size, d_model)
        self.sigma = nn.Linear(rnn_hidden_size, d_model)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  
        mu = self.mu(rnn_out)
        sigma = torch.exp(self.sigma(rnn_out)) 
        return mu, sigma


class HawkesIntensityFunction(nn.Module):
    def __init__(self, d_model):
        super(HawkesIntensityFunction, self).__init__()
        self.mu = nn.Parameter(torch.ones(1))  
        self.alpha = nn.Parameter(torch.ones(1))  
        self.beta = nn.Parameter(torch.ones(1))  
        self.gamma = nn.Parameter(torch.ones(1))  

        self.intensity_layer = nn.Linear(d_model, 1)

    def forward(self, z, event_times, event_locations):

        base_intensity = torch.exp(self.intensity_layer(z))
        event_times = event_times.unsqueeze(-1)
        time_diffs = event_times - event_times.transpose(1, 2)
        time_diffs = torch.triu(time_diffs, diagonal=1)
        excitation_temporal = self.alpha * torch.exp(-self.beta * time_diffs)
        event_locations = event_locations.unsqueeze(-2)
        spatial_diffs = event_locations - event_locations.transpose(1, 2)
        spatial_distances = torch.norm(spatial_diffs, dim=-1)
        spatial_diffs = torch.triu(spatial_distances, diagonal=1)
        excitation_spatial = torch.exp(-self.gamma * spatial_diffs)
        excitation = excitation_temporal * excitation_spatial
        excitation_sum = excitation.sum(dim=2, keepdim=True)
        intensity = self.mu + base_intensity + excitation_sum
        return intensity
    
    def log_likelihood(self, z, event_times, event_locations):
        intensity = self.forward(z, event_times, event_locations)
        log_intensity = torch.log(intensity + 1e-9)
        event_indices = event_times.unsqueeze(-1).long()
        event_indices = torch.clamp(event_indices, 0, intensity.size(1) - 1)
        log_intensity_values = log_intensity.gather(1, event_indices)
        integrated_intensity = torch.cumsum(intensity, dim=1)
        log_likelihood = log_intensity_values.sum(dim=1) - integrated_intensity.sum(dim=1)
        return log_likelihood.mean()
    
    
    
    def single_event_intensity(self, z, event_times, event_locations, predicted_time, predicted_location):
        base_intensity = torch.exp(self.intensity_layer(z[:, -1, :]))
        predicted_time = predicted_time.unsqueeze(-1)
        time_diffs = predicted_time - event_times
        excitation_temporal = self.alpha * torch.exp(-self.beta * time_diffs)
        predicted_location = predicted_location.unsqueeze(1)
        spatial_diffs = predicted_location - event_locations
        spatial_distances = torch.norm(spatial_diffs, dim=-1)
        excitation_spatial = torch.exp(-self.gamma * spatial_distances)
        
        excitation = excitation_temporal * excitation_spatial
        excitation_sum = excitation.sum(dim=1, keepdim=True)
        intensity = torch.exp(self.mu) + base_intensity + excitation_sum
        return intensity


    def generate_event_thinning(self, z, event_times, event_locations, T):
        batch_size = event_times.shape[0]
        all_predicted_times = []
        all_predicted_locations = []
        for i in range(batch_size):
            current_time = event_times[i, -1].item()
            current_location = event_locations[i, -1, :]
            times = event_times[i].tolist()
            locations = event_locations[i].tolist()

            predicted_times = []
            predicted_locations = []

            accepted_event = False

            while not accepted_event:
                upper_bound = self.single_event_intensity(z[i:i+1], event_times[i:i+1], event_locations[i:i+1], torch.tensor([current_time]), current_location.unsqueeze(0)).max().item()
                inter_event_time = torch.distributions.Exponential(upper_bound).sample().item()
                candidate_time = current_time + inter_event_time
                if candidate_time > T[i].item():
                    break

                candidate_location = torch.distributions.Normal(current_location, 1.0).sample().squeeze(0)
                candidate_intensity = self.single_event_intensity(z[i:i+1], torch.tensor(times).unsqueeze(0), torch.tensor(locations).unsqueeze(0), torch.tensor([candidate_time]), candidate_location.unsqueeze(0))
                acceptance_prob = candidate_intensity / upper_bound

                if torch.rand(1).item() < acceptance_prob:
                    predicted_times.append(candidate_time)
                    predicted_locations.append(candidate_location)
                    current_time = candidate_time
                    current_location = candidate_location
                    accepted_event = True

            if len(predicted_times) == 0:
                candidate_time = current_time + torch.distributions.Exponential(upper_bound).sample().item()
                candidate_location = torch.distributions.Normal(current_location, 1.0).sample().squeeze(0)
                predicted_times.append(candidate_time)
                predicted_locations.append(candidate_location)
                current_time = candidate_time
                current_location = candidate_location
                times.append(candidate_time)
                locations.append(candidate_location.tolist())

            all_predicted_times.append(predicted_times)
            all_predicted_locations.append(torch.stack(predicted_locations))

        max_length = max(len(pt) for pt in all_predicted_times)
        padded_times = [pt + [0] * (max_length - len(pt)) for pt in all_predicted_times]
        all_predicted_times_tensor = torch.tensor(padded_times)

        all_predicted_locations_tensor = torch.cat(all_predicted_locations, dim=0)
        return all_predicted_times_tensor, all_predicted_locations_tensor




   
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
    
class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, input_dim) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, input_dim)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, input_dim)
        return self.proj(x)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, 
                 src_pos: PositionEncoding, tgt_pos: PositionEncoding, projection_layer: ProjectionLayer,
                 d_model: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.latent_process = LatentProcess(d_model)
        self.intensity_function = HawkesIntensityFunction(d_model)

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    def forward(self, src, src_mask, T):
        # Encoding
        encoder_output = self.encode(src, src_mask)
        
        # Latent process
        latent_mu, latent_sigma = self.latent_process(encoder_output)
        z = latent_mu + torch.randn_like(latent_mu) * latent_sigma
        
        # Get the event times and locations from the source sequence
        event_times = src[:, :, -1]
        event_locations = src[:, :, :-1]
        
        # Generate the next event's time and location using the thinning algorithm
        predicted_time, predicted_location = self.intensity_function.generate_event_thinning(z, event_times, event_locations, T)
        # Combine predicted time and location
        predicted_event = torch.cat([predicted_location, predicted_time], dim=-1)
        
        return predicted_event, z
    
    def log_likelihood(self, z, event_times, event_locations):
        return self.intensity_function.log_likelihood(z, event_times, event_locations)

    
def build_transformer(src_input_dim: int, tgt_input_dim: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_input_dim)
    tgt_embed = InputEmbedding(d_model, tgt_input_dim)

    # Create the positional encoding layers
    src_pos = PositionEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_input_dim)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer, d_model)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer