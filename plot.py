import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation

def plot_lambst_static(all_intensities, accepted_events, rejected_events, x_range, y_range, t_range, fps, 
                       cmap='magma', fn='result.mp4', transition_frames=10):
    cmin = 0
    cmax = max(np.max(batch) for batch in all_intensities)
    cmid = cmin + (cmax - cmin) * 0.9    
    print(f'Inferred cmax: {cmax}')
        
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    batch_count = len(all_intensities)
    frames_per_batch = len(t_range)  # frame number of the animation per batch
    total_frames = batch_count * (frames_per_batch + transition_frames * (batch_count - 1))

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d', xlabel='x', ylabel='y', zlabel='λ', zlim=(cmin, cmax), 
                         title='Spatio-temporal Conditional Intensity')
    ax.title.set_position([.5, .95])
    ax.set_xlim(min(x_range), max(x_range))  # Explicitly set x-axis limits
    ax.set_ylim(min(y_range), max(y_range))
    ax.set_zlim(cmin, cmax)
    cumulative_time = 0  # Initialize cumulative time
    text = ax.text(min(x_range), min(y_range), cmax, "t={:.2f}".format(t_range[0]), fontsize=10)
    plot = [ax.plot_surface(grid_x, grid_y, all_intensities[0][0], rstride=1, cstride=1, cmap=cmap)]
    
    zs = np.ones_like(all_intensities[0][0]) * cmid  # add a platform for locations
    plat = ax.plot_surface(grid_x, grid_y, zs, rstride=1, cstride=1, color='white', alpha=0.2)
    accepted_points = ax.scatter3D([], [], [], color='green', label='Accepted')  # add locations for accepted events
    rejected_points = ax.scatter3D([], [], [], color='red', label='Rejected')  # add locations for rejected events
    plot.append(plat)
    plot.append(accepted_points)
    plot.append(rejected_points)    
    
    pbar = tqdm(total=total_frames + 2)
    
    def update_plot(frame_number):
        nonlocal cumulative_time
        batch_index = frame_number // (frames_per_batch + transition_frames)
        frame_in_batch = frame_number % (frames_per_batch + transition_frames)

        if frame_in_batch < frames_per_batch:
            t = t_range[frame_in_batch]
            current_intensity = all_intensities[batch_index][frame_in_batch]
        else:
            # Interpolate between the last frame of the current batch and the first frame of the next batch
            next_batch_index = min(batch_index + 1, batch_count - 1)
            alpha = (frame_in_batch - frames_per_batch) / transition_frames
            current_intensity = (1 - alpha) * all_intensities[batch_index][-1] + alpha * all_intensities[next_batch_index][0]
            t = t_range[-1] + alpha * (t_range[1] - t_range[0])  # Adjust time for smooth transition

        cumulative_time += t_range[1] - t_range[0] if frame_in_batch < frames_per_batch else 0  # Accumulate time
        plot[0].remove()
        plot[0] = ax.plot_surface(grid_x, grid_y, current_intensity, rstride=1, cstride=1, cmap='magma')
        text.set_text('t={:.2f}'.format(cumulative_time))  # Use cumulative time
                
        # Update the scatter plots for accepted and rejected events without removing the previous ones
        accepted_locs = np.array([event[:2] for event in accepted_events if event[2] <= cumulative_time])
        rejected_locs = np.array([event[:2] for event in rejected_events if event[2] <= cumulative_time])
        
        if accepted_locs.size > 0:
            zs = np.ones_like(accepted_locs[:, 0]) * cmid
            plot[2]._offsets3d = (accepted_locs[:, 0], accepted_locs[:, 1], zs)
        
        if rejected_locs.size > 0:
            zs = np.ones_like(rejected_locs[:, 0]) * cmid
            plot[3]._offsets3d = (rejected_locs[:, 0], rejected_locs[:, 1], zs)
        
        ax.set_xlim(min(x_range), max(x_range))  # Reset x-axis limits
        ax.set_ylim(min(y_range), max(y_range))
        ax.set_zlim(cmin, cmax)
        pbar.update()
    
    ani = animation.FuncAnimation(fig, update_plot, total_frames, interval=1000/fps)
    ani.save(fn, writer='ffmpeg', fps=fps)
    return ani
