import numpy as np
from scipy.interpolate import CubicSpline
import random

def generate_fixed_speed_schedule(n_timesteps):
    return np.ones((n_timesteps, 1)) * 1.0

def generate_random_speed_schedule(n_timesteps):
    speed_schedule = np.zeros((n_timesteps, 1)) 
    # Generate a random sine wave combination 
    ts = np.linspace(0, 1, n_timesteps)
    for i in range(5):
        a = np.random.uniform() 
        phi = np.random.uniform() * 2 * np.pi 
        omega = np.random.uniform(low = 0.1, high = 20)
        speed_schedule[:,0] += a * np.sin(omega * ts + phi)
    return speed_schedule

def generate_sigmoid_speed_schedule(n_timesteps):

    # vel1 = np.array([random.random(), ] * n_timesteps)
    vel1 = np.array([0.35, ] * n_timesteps)


    # vel2 = np.array([np.round(random.random(), decimals =1),] * self.data_size)
    # vel2 = np.array([random.random(),] * n_timesteps)
    vel2 = np.array([0.65, ] * n_timesteps)

    if vel1 is vel2:
        vel2 =[np.array(random.random()), ] * n_timesteps

    w = 0.15
    D = np.linspace(0, 2, n_timesteps)
    sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
    vels = vel1 + (vel2 - vel1) * (1 - sigmaD)

    velocity = np.zeros((n_timesteps, 3))
    velocity[:, 0] = vels
    velocity[:,1] = np.zeros_like(vels)
    velocity[:, 2] = np.zeros_like(vels)

    return velocity


def generate_sigmoid_velocity_steps(n_timesteps):
    # Change step for smaller or larger transisitons
    # vel1 = np.array([random.random(), ] * n_timesteps)
    vel1 = np.array([0.35, ] * n_timesteps)
    vel2 = np.array([0.65, ] * n_timesteps)

    #  vel2 = np.array([np.round(random.random(), decimals =1),] * self.data_size)
    # # vel2 = np.array([random.random(),] * n_timest

    data_size=n_timesteps

    num_steps = int(100 * abs(vel1[0] - vel2[0])) + 1
    vels_met = np.linspace(vel1[0], vel2[0], num_steps)
    vel_range = int(100 * abs(vels_met[0] - vels_met[-1]))

    if vel_range != 0:
        data_size_interval = int(n_timesteps/ vel_range)
    else:
        data_size_interval = 1

    vals = []
    i = 0

    if data_size_interval * vel_range != data_size:
        diff = n_timesteps - data_size_interval * vel_range
        if diff > 0:
            data_size_interval += diff
        else:
            data_size_interval -= diff

    data_size_interval = data_size_interval

    while i <= len(vels_met) - 2:
        w = 0.1
        D = np.linspace(0, 2, data_size_interval)
        sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
        val = vels_met[i] + (vels_met[i + 1] - vels_met[i]) * (1 - sigmaD)
        vals.append(val)
        i = i + 1

    vals = np.array(vals)
    velocity_profile = vals.flatten()

    if velocity_profile.size != data_size:
        diff = velocity_profile.size - data_size
        if diff > 0:
            velocity_profile = velocity_profile[:velocity_profile.size - diff]


    velocity = np.zeros((n_timesteps, 3))
    velocity[:, 0] = velocity_profile
    velocity[:,1] = np.zeros_like(velocity[:, 0])
    velocity[:, 2] = np.zeros_like(velocity[:, 0])

    return velocity





def generate_random_polar_direction_schedule(n_timesteps):
    direction_schedule = np.zeros((n_timesteps, 3))
    ts = np.linspace(0, 1, n_timesteps)

    theta = np.zeros(n_timesteps)
    for i in range(5):
        phi = np.random.uniform() * 2 * np.pi 
        omega = np.random.uniform(low = 0.1, high = 20)
        theta += np.sin(omega * ts + phi)

    direction_schedule[:, 0] += np.cos(theta)
    direction_schedule[:, 1] += np.sin(theta)
    return direction_schedule

def generate_cubic_spline_direction_schedule(n_timesteps):
    direction_schedule = np.zeros((n_timesteps, 2))

    xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # Select random direction
    ys = np.random.normal(loc=np.zeros((xs.shape[0], 2)), scale=np.ones((xs.shape[0], 2)))
    ys = ys / np.linalg.norm(ys, axis=-1, keepdims=True)
    
    # Interpolate to fill in schedule
    cs = CubicSpline(xs, ys)
    direction_schedule += cs(ts, 1)
    return direction_schedule

if __name__ == "__main__":
    
    n_timesteps = 10000
    ts = np.linspace(0, 20, n_timesteps)

    # speed_schedule = generate_random_speed_schedule(n_timesteps)
    # direction_schedule = generate_random_polar_direction_schedule(n_timesteps)
    # velocity_schedule = speed_schedule * direction_schedule
    # combined_schedule = np.concatenate([velocity_schedule, speed_schedule], axis=-1)


    velocity_schedule = generate_sigmoid_speed_schedule(n_timesteps)
    speed = velocity_schedule[:,0]
    speed_schedule = np.reshape(speed, (len(velocity_schedule[:,0]), 1))

    # Visualize the schedule
    import matplotlib.pyplot as plt 
    plt.plot(ts, velocity_schedule)
    plt.show()
    combined_schedule= np.concatenate([velocity_schedule, speed_schedule], axis=-1)

    inp = input("[S]ave, or [E]xit\n")
    if inp.lower() == "s":
        np.save('../schedule.npy', combined_schedule)