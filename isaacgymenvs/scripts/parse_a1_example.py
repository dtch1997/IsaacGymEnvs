import argparse
import isaacgym
import torch
import pathlib
from isaacgymenvs.utilities.quadruped_motion_data import MotionLib

from typing import Dict, Tuple
import numpy as np 
import json 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert a1_expert_raw to desired format")
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("-s", "--start-time-frac", type=float, help="Start time as a fraction. E.g. 0.2 = start from 20% of the way in", default = 0.0)
    parser.add_argument("-e", "--end-time-frac", type=float, help="End time as a fraction. E.g. 0.8 = end at 80% of the way in", default = 1.0)
    args = parser.parse_args()
    return args

def read_csv(filepath: str, delimiter=',') -> np.ndarray:
    return np.genfromtxt(filepath, delimiter=delimiter)

def reorder_dofs(dof_data: np.ndarray) -> np.ndarray:
    """
    Input: FR_HAA_Pos,FR_HFE_Pos,FR_KFE_Pos,FL_HAA_Pos,FL_HFE_Pos,FL_KFE_Pos,RR_HAA_Pos,RR_HRE_Pos,RR_KRE_Pos,RL_HAA_Pos,RL_HRE_Pos,RL_KRE_Pos
    Output: ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    
    Input: (n_dof, ...)
    Output: (n_dof, ...)
    """
    fr_data = dof_data[:3, ...]
    fl_data = dof_data[3:6, ...]
    rr_data = dof_data[6:9, ...]
    rl_data = dof_data[9:12, ...]
    return np.concatenate([fl_data, fr_data, rl_data, rr_data], axis=0)

def parse_mocap_data(filepath: str) -> Tuple[np.ndarray, float]:
    """ 
    Input: Filepath to raw data 
    Output: (T, 19) array, each row is (body_pos, body_orn, dof_pos)
    """
    arr = read_csv(filepath)
    time = arr[:,0]
    dt = time[1] - time[0] # assume constant timestep
    body_pos = arr[:,1:4] # x, y, z
    body_orn = arr[:,4:8] # x, y, z, w
    dof_pos = reorder_dofs(arr[:,8:20].T).T # (FL, FR, RL, RR) sequence of (hip, thigh, calf) 
    frame_data = np.concatenate([body_pos, body_orn, dof_pos], axis=-1)
    return frame_data, dt

def write_motion_data(filepath: str, frames: np.ndarray, dt: float, loop_mode: str = 'Clamp', enable_cycle_offset_position: bool = True, enable_cycle_offset_rotation: bool = False):
    motion_data = {}
    motion_data["LoopMode"] = loop_mode
    motion_data["Frames"] = frames.tolist()
    motion_data["FrameDuration"] = dt
    motion_data["EnableCycleOffsetPosition"] = enable_cycle_offset_position
    motion_data["EnableCycleOffsetRotation"] = enable_cycle_offset_rotation
    with open(filepath, 'w') as file:
        json.dump(motion_data, file)

if __name__ == "__main__":
    args = parse_args()

    input_dir = pathlib.Path(args.input_dir)
    filepaths = input_dir.rglob("*.csv")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    output_filenames = []

    for filepath in filepaths:
        count += 1
        frame_data, dt = parse_mocap_data(str(filepath))
        n_timesteps = frame_data.shape[0]
        start_time = int(n_timesteps * args.start_time_frac)
        end_time = int(n_timesteps * args.end_time_frac)
        
        output_filename = filepath.relative_to(input_dir)
        output_filename = output_filename.parent / output_filename.stem
        output_filename = '_'.join(str(output_filename).split('/'))
        output_filename += '.txt'

        # Ensure no repeating filenames
        assert output_filename not in output_filenames
        output_filenames.append(output_filename)

        write_motion_data(
            output_dir / (output_filename), 
            frame_data[start_time: end_time], 
            dt, loop_mode='Clamp'
        )

    # Write metadata
    metadatas = []
    for fp in output_filenames:
        metadata = {'file': fp, 'weight': 1 / count}
        metadatas.append(metadata)
    import yaml
    with open(output_dir / 'dataset.yaml', 'w') as dataset_file:
        yaml.dump({'motions': metadatas}, dataset_file)

    