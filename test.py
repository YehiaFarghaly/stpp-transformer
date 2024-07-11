import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
model.load_state_dict(torch.load('best_transformer_model2.pth'))
model.eval()


example_events = pd.DataFrame({
    'x': [37.45401188, 95.07143064, 73.19939418, 59.86584842, 15.60186404, 15.59945203, 5.808361217, 86.61761458, 60.11150117, 70.80725778],
    'y': [18.51329288, 54.19009474, 87.29458359, 73.22248864, 80.65611479, 65.87833667, 69.22765645, 84.91956516, 24.96680089, 48.94249636],
    'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})


example_tensor = torch.tensor(example_events.values, dtype=torch.float32).unsqueeze(0)


with torch.no_grad():
    src_mask = None
    tgt_mask = None
    predictions, _ = model(example_tensor, example_tensor, src_mask, tgt_mask)

predictions_np = predictions.numpy()

predictions_df = pd.DataFrame(predictions_np, columns=['x', 'y', 'time'])

max_time = int(max(example_events['time'].max() + 1, predictions_df['time'].max() + 1))
time_step = 0.1
fig, ax = plt.subplots()
ax.set_xlim(min(example_events['x']) - 1, 100)
ax.set_ylim(min(example_events['y']) - 1, 100)

input_scatter = ax.scatter([], [], c='blue', label='Input Events')
predicted_scatter = ax.scatter([], [], c='red', label='Predicted Events')

def update(frame):
    current_time = frame * time_step

    input_data = example_events[example_events['time'] <= current_time]
    predicted_data = predictions_df[predictions_df['time'] <= current_time]
    input_scatter.set_offsets(input_data[['x', 'y']].values)
    predicted_scatter.set_offsets(predicted_data[['x', 'y']].values)
    return input_scatter, predicted_scatter

ani = animation.FuncAnimation(fig, update, frames=range(int(max_time / time_step) + 1), interval=100, blit=True, repeat=True)


plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Spatiotemporal Events')
plt.show()
