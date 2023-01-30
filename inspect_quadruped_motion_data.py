import isaacgym 
import torch 
from isaacgymenvs.tasks.quadruped_motion_data import MotionLib 

import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", type=str)
    parser.add_argument("--output-file", type=str, default="")
    args = parser.parse_args()
    return args

def visualize(motion_lib: MotionLib, save_path: str = ""):

    start_time = 0
    end_time = motion_lib.get_total_length()
    num_interp = 1000

    motion_ids = np.zeros(num_interp, dtype=np.int32)
    motion_times = np.linspace(start_time, end_time, num = num_interp)
    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel = motion_lib.get_motion_state(motion_ids, motion_times)

    fig, ax = plt.subplots(2,2, figsize=(20, 20))

    # Plot body pos
    assert root_pos.shape[1] == 3
    for i in range(root_pos.shape[1]):
        ax[0][0].plot(motion_times, root_pos[:, i])
    ax[0][0].set_title("Root Pos")

    # Plot body rot
    assert root_rot.shape[1] == 4
    for i in range(root_rot.shape[1]):
        ax[1][0].plot(motion_times, root_rot[:, i])
    ax[1][0].set_title("Root Rot")

    # Plot dof pos
    assert dof_pos.shape[1] == 12
    for i in range(dof_pos.shape[1]):
        ax[0][1].plot(motion_times, dof_pos[:, i])
    ax[0][1].set_title("Dof Pos")

    # Plot dof vel
    assert dof_vel.shape[1] == 12
    for i in range(dof_vel.shape[1]):
        ax[1][1].plot(motion_times, dof_vel[:, i])
    ax[0][1].set_title("Dof Vel")
    
    plt.show()
    if save_path:
        plt.savefig(save_path)

if __name__ == "__main__":
    try:
        args = parse_args()
        device = torch.device('cpu')
        motion_lib = MotionLib(args.motion_file, device)
        visualize(motion_lib, args.output_file)
    except KeyboardInterrupt:
        pass