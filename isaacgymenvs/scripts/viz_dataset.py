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

def plot_motion_data(ts, root_states, dof_states) -> plt.Axes:
    fig, ax = plt.subplots(2, 3, figsize=(40, 20))

    # Body pos 
    body_pos = root_states[:, :3]
    line_plot(ax[0][0], ts, body_pos, title="Body Pos")

    # Body orn
    body_orn = root_states[:, 3:7]
    line_plot(ax[0][1], ts, body_orn, title="Body Orn")

    # Dof pos
    dof_pos = dof_states[:, :12]
    line_plot(ax[0][2], ts, dof_pos, title="Dof Pos")

    # Body ang vel
    body_lin_vel = root_states[:, 10:]
    line_plot(ax[1][0], ts, body_lin_vel, title="Body Lin Vel")

    # Body ang vel
    body_ang_vel = root_states[:, 3:6]
    line_plot(ax[1][1], ts, body_ang_vel, title="Body Ang Vel")

    # Dof vel
    dof_vel = dof_states[:, 12:]
    line_plot(ax[1][2], ts, dof_vel, title="Dof Vel") 

    return fig, ax 

if __name__ == "__main__":

    args = parse_args()
    dataset = h5py.File(args.filepath)
    for name in dataset.keys():
        print(name)
        print(dataset[name].shape)
        print(dataset[name].attrs['size'])

    ts = np.arange(200) * 0.02
    root_states = dataset['root_states'][0]
    dof_states = dataset['dof_states'][0]
    fig, ax = plot_motion_data(ts, root_states, dof_states)
    filename = pathlib.Path(args.filepath).stem
    fig.suptitle(filename, size=30)
    fig.show()    
    input("Press any key to exit...")