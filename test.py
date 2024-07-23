import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
from model import build_transformer

# Initialize model parameters
src_input_dim = 3
tgt_input_dim = 3
src_seq_len = 10
tgt_seq_len = 1
d_model = 512
N = 6
h = 8
dropout = 0.1
d_ff = 2048

# Load model
model = build_transformer(src_input_dim, tgt_input_dim, src_seq_len, tgt_seq_len, d_model, N, h, dropout, d_ff)
model.load_state_dict(torch.load('best_transformer_model.pth'))
print('model loaded')
model.eval()

# Example events data
example_events = pd.DataFrame({
    'x': [37.45401188, 95.07143064, 73.19939418, 59.86584842, 15.60186404, 15.59945203, 5.808361217, 86.61761458, 60.11150117, 70.80725778],
    'y': [18.51329288, 54.19009474, 87.29458359, 73.22248864, 80.65611479, 65.87833667, 69.22765645, 84.91956516, 24.96680089, 48.94249636],
    'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

# Convert to tensor
example_tensor = torch.tensor(example_events.values, dtype=torch.float32).unsqueeze(0)
time_tensor = torch.tensor([8, 9, 10, 7, 8, 7.6, 8.4, 6.9, 9.6, 10])
# Model prediction
with torch.no_grad():
    src_mask = None
    tgt_mask = None
    print('inside the loop')
    predictions, z = model(example_tensor, src_mask, time_tensor)
print('predictions = ', predictions)
predictions_np = predictions.numpy()
predictions_df = pd.DataFrame(predictions_np, columns=['x', 'y', 'time'])

# Get the intensity function values
event_times = torch.tensor(example_events['time'].values, dtype=torch.float32).unsqueeze(0)
event_locations = torch.tensor(example_events[['x', 'y']].values, dtype=torch.float32).unsqueeze(0)
predicted_time = torch.tensor(predictions_df['time'].values, dtype=torch.float32)
predicted_location = torch.tensor(predictions_df[['x', 'y']].values, dtype=torch.float32)

print('predicted: ', predictions_df )
with torch.no_grad():
    input_intensity = model.intensity_function(z, event_times, event_locations).squeeze().numpy()
    predicted_intensity = model.intensity_function.single_event_intensity(z, event_times, event_locations, predicted_time, predicted_location).squeeze().numpy()
    print('inten: ', predicted_intensity)
intensity_df = pd.DataFrame({
    'x': example_events['x'].tolist() + predictions_df['x'].tolist(),
    'y': example_events['y'].tolist() + predictions_df['y'].tolist(),
    'intensity': input_intensity.tolist() + [predicted_intensity],
    'time': example_events['time'].tolist() + predictions_df['time'].tolist(),
    'type': ['input'] * len(example_events) + ['predicted'] * len(predictions_df)
})

print(intensity_df)

# Define a grid for x and y
x = np.linspace(min(intensity_df['x']), max(intensity_df['x']), 100)
y = np.linspace(min(intensity_df['y']), max(intensity_df['y']), 100)
x_grid, y_grid = np.meshgrid(x, y)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(min(intensity_df['x']), max(intensity_df['x']))
ax.set_ylim(min(intensity_df['y']), max(intensity_df['y']))
ax.set_zlim(0, 0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensity')
plt.title('Spatio-temporal Conditional Intensity')

# Initialize scatter plots
input_scatter = ax.scatter([], [], [], c='blue', label='Input Events')
predicted_scatter = ax.scatter([], [], [], c='red', label='Predicted Events')

# Animation update function
def update(frame):
    current_time = frame * time_step
    ax.clear()

    # Interpolate intensity values over the grid
    current_intensity_df = intensity_df[intensity_df['time'] <= current_time]
    decay_factor = np.exp(-0.2 * (current_time - intensity_df['time']))
    decayed_intensity = intensity_df['intensity'] * decay_factor
    if len(current_intensity_df) >= 3:
        z = griddata(
            (current_intensity_df['x'], current_intensity_df['y']),
            decayed_intensity[intensity_df['time'] <= current_time],
            (x_grid, y_grid),
            method='cubic',
            fill_value=0
        )
        ax.plot_surface(x_grid, y_grid, z, cmap=cm.viridis, alpha=0.6)
    
    # Update scatter plots
    input_data = intensity_df[(intensity_df['type'] == 'input') & (intensity_df['time'] <= current_time)]
    predicted_data = intensity_df[(intensity_df['type'] == 'predicted') & (intensity_df['time'] <= current_time)]
    input_scatter._offsets3d = (input_data['x'].values, input_data['y'].values, input_data['intensity'].values)
    predicted_scatter._offsets3d = (predicted_data['x'].values, predicted_data['y'].values, predicted_data['intensity'].values)
    
    ax.scatter(input_data['x'].values, input_data['y'].values, input_data['intensity'].values, c='blue', label='Input Events')
    ax.scatter(predicted_data['x'].values, predicted_data['y'].values, predicted_data['intensity'].values, c='red', label='Predicted Events')
    
    ax.set_xlim(min(intensity_df['x']), max(intensity_df['x']))
    ax.set_ylim(min(intensity_df['y']), max(intensity_df['y']))
    ax.set_zlim(0, max(intensity_df['intensity'] + 0.2))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    plt.title('Spatio-temporal Conditional Intensity')
    ax.text2D(0.05, 0.95, f"t={current_time:.2f}", transform=ax.transAxes)

# Set up animation
max_time = max(intensity_df['time']) + 1
time_step = 0.1
ani = animation.FuncAnimation(fig, update, frames=range(int(max_time / time_step) + 1), interval=100, blit=False, repeat=True)

plt.legend()
plt.show()
