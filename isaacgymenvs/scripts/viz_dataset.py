import matplotlib.pyplot as plt
import argparse
import h5py
import numpy as np
import pathlib

from typing import Optional, List

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize expert dataset")
    parser.add_argument("-f", "--filepath", type=str)
    args = parser.parse_args()
    return args

def line_plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, 
              title: str, fontsize: int = 20,
              xlabel: str = "", 
              ylabel: str = "", 
              labels: Optional[List[str]] = None):
    ax.plot(x, y)
    ax.set_title(title, size= fontsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels is not None:
        ax.legend(labels)

def plot_motion_data(ts, root_states, dof_pos, dof_vel) -> plt.Axes:
    fig, ax = plt.subplots(2, 3, figsize=(40, 20))
    fig.set_tight_layout(True)

    # Body pos 
    body_pos = root_states[:, :3]
    line_plot(ax[0][0], ts, body_pos, 
            title="Body Pos", 
            xlabel="Time (s)",
            ylabel="Position (m)",
            labels = ["x", "y", "z"]
    )

    # Body orn
    body_orn = root_states[:, 3:7]
    line_plot(ax[0][1], ts, body_orn, 
            title="Body Orn",  
            xlabel="Time (s)", 
            labels = ["qx", "qy", "qz", "qw"]
    )

    # Dof pos
    line_plot(ax[0][2], ts, dof_pos, 
            title="Dof Pos",  
            xlabel="Time (s)",
            ylabel="Joint pos (rad)"
    )

    # Body ang vel
    body_lin_vel = root_states[:, 7:10]
    line_plot(ax[1][0], ts, body_lin_vel, 
            title="Body Lin Vel",  
            xlabel="Time (s)",
            ylabel="Velocity (m/s)",
            labels=["dx", "dy", "dz"]
    )

    # Body ang vel
    body_ang_vel = root_states[:, 10:]
    line_plot(ax[1][1], ts, body_ang_vel, 
            title="Body Ang Vel",  
            xlabel="Time (s)", 
            ylabel="Ang vel (rad/s)",
            labels=["dR", "dP", "dY"]
    )

    # Dof vel
    line_plot(ax[1][2], ts, dof_vel, 
            title="Dof Vel",  
            xlabel="Time (s)",
            ylabel="Joint vel (rad/s)"
              
    ) 

    return fig, ax 

if __name__ == "__main__":

    args = parse_args()
    file = h5py.File(args.filepath)
    for key, value in file.attrs.items():
        print(f"{key}: {value}")
    for name in file.keys():
        print(name)
        print(file[name].shape)
        print(file[name].attrs['size'])

    ts = np.arange(file.attrs['max_episode_length']) * file.attrs['dt']
    root_states = file['root_states'][0]
    dof_pos = file['dof_pos'][0]
    dof_vel = file['dof_vel'][0]
    fig, ax = plot_motion_data(ts, root_states, dof_pos, dof_vel)
    filename = pathlib.Path(args.filepath).stem
    fig.suptitle(filename, size=30)
    fig.show()    
    input("Press any key to exit...")