import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation

def plot_lambst_static(intensities, accepted_events, rejected_events, x_range, y_range, t_range, fps, 
                         cmap='magma', fn='result.mp4'):        
    cmin = 0
    cmax = np.max(intensities)
        
    cmid = cmin + (cmax - cmin) * 0.9    
        
    print(f'Inferred cmax: {cmax}')
        
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    frn = len(t_range)  # frame number of the animation

    fig = plt.figure(figsize=(6,6), dpi=150)
    ax = fig.add_subplot(111, projection='3d', xlabel='x', ylabel='y', zlabel='λ', zlim=(cmin, cmax), 
                         title='Spatio-temporal Conditional Intensity')
    ax.title.set_position([.5, .95])
    ax.set_xlim(min(x_range), max(x_range))  # Explicitly set x-axis limits
    ax.set_ylim(min(y_range), max(y_range))
    ax.set_zlim(cmin, cmax)
    text = ax.text(min(x_range), min(y_range), cmax, "t={:.2f}".format(t_range[0]), fontsize=10)
    plot = [ax.plot_surface(grid_x, grid_y, intensities[0], rstride=1, cstride=1, cmap=cmap)]
    
    zs = np.ones_like(intensities[0]) * cmid # add a platform for locations
    plat = ax.plot_surface(grid_x, grid_y, zs, rstride=1, cstride=1, color='white', alpha=0.2)
    accepted_points = ax.scatter3D([], [], [], color='green', label='Accepted') # add locations for accepted events
    rejected_points = ax.scatter3D([], [], [], color='red', label='Rejected') # add locations for rejected events
    plot.append(plat)
    plot.append(accepted_points)
    plot.append(rejected_points)    
    
    pbar = tqdm(total=frn + 2)
    
    def update_plot(frame_number):
        t = t_range[frame_number]
        plot[0].remove()
        plot[0] = ax.plot_surface(grid_x, grid_y, intensities[frame_number], rstride=1, cstride=1, cmap='magma')
        text.set_text('t={:.2f}'.format(t))
                
        # Update the scatter plots for accepted and rejected events
        accepted_locs = np.array([event[:2] for event in accepted_events if event[2] <= t])
        rejected_locs = np.array([event[:2] for event in rejected_events if event[2] <= t])
        
        if accepted_locs.size > 0:
            zs = np.ones_like(accepted_locs[:, 0]) * cmid
            plot[2].remove()
            plot[2] = ax.scatter3D(accepted_locs[:, 0], accepted_locs[:, 1], zs, color='green', s=10, label='Accepted')
        
        if rejected_locs.size > 0:
            zs = np.ones_like(rejected_locs[:, 0]) * cmid
            plot[3].remove()
            plot[3] = ax.scatter3D(rejected_locs[:, 0], rejected_locs[:, 1], zs, color='red', s=10, label='Rejected')
        
        ax.set_xlim(min(x_range), max(x_range))  # Reset x-axis limits
        ax.set_ylim(min(y_range), max(y_range))
        ax.set_zlim(cmin, cmax)
        pbar.update()
    
    ani = animation.FuncAnimation(fig, update_plot, frn, interval=1000/fps)
    ani.save(fn, writer='ffmpeg', fps=fps)
    return ani
