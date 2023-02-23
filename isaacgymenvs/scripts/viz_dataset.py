import matplotlib.pyplot as plt
import argparse
import h5py
import numpy as np
import pathlib

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize expert dataset")
    parser.add_argument("--filepath", type=str)
    args = parser.parse_args()
    return args

def line_plot(ax: plt.Axes, x: np.ndarray, y: np.ndarray, title: str, fontsize: int = 20):
    ax.plot(x, y)
    ax.set_title(title, size= fontsize)

def plot_motion_data(ts, root_states) -> plt.Axes:
    fig, ax = plt.subplots(2, 2, figsize=(30, 20))

    # Body pos 
    body_pos = root_states[:, :3]
    line_plot(ax[0][0], ts, body_pos, title="Body Pos")

    # Body orn
    body_orn = root_states[:, 3:7]
    line_plot(ax[0][1], ts, body_orn, title="Body Orn")

    # Dof pos
    # dof_pos = root_states[:, 7:10]
    # line_plot(ax[0][2], ts, dof_pos, title="Dof Pos")

    # Body ang vel
    body_lin_vel = root_states[:, 10:]
    line_plot(ax[1][0], ts, body_lin_vel, title="Body Lin Vel")

    # Body ang vel
    body_ang_vel = root_states[:, 3:6]
    line_plot(ax[1][1], ts, body_ang_vel, title="Body Ang Vel")

    # Dof vel
    # dof_vel = frame_vels[:, 6:]
    # line_plot(ax[1][2], ts, dof_vel, title="Dof Vel") 

    return fig, ax 

if __name__ == "__main__":

    args = parse_args()
    dataset = h5py.File(args.filepath)
    print(dataset['root_states'].shape)
    print(dataset['root_states'].attrs['size'])
    print(dataset['actions'].shape)
    print(dataset['actions'].attrs['size'])

    ts = np.arange(200) * 0.02
    fig, ax = plot_motion_data(ts, dataset['root_states'][0])
    filename = pathlib.Path(args.filepath).stem
    fig.suptitle(filename, size=30)
    fig.show()    
    input("Press any key to exit...")