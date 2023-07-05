import argparse
import h5py
import numpy as np
import os
import pathlib
import torch
import csv
from tabulate import tabulate
import matplotlib.pyplot as plt



class Evaluation:
    def __init__(self,path,num_files,vels, max_time):

        self.ref_path = path
        self.num_files = num_files
        self.path_folder = os.path.dirname(os.getcwd())
        self._velocity = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        self._plot_all_data = False
        self._plot_all_errors = True
        self._plot_single_data = False

        self._desired_vel = int(10*vels)
        self.max_time =max_time

        if type(self._velocity) == float:
            self.velocity_index = int(10 * self._velocity)
        else:
            self.velocity_index = [int(10*item) for item in self._velocity]



    def load_target_data(self):


        file_dir = pathlib.Path(self.ref_path)
        folder_dir = pathlib.Path(self.path_folder)

        input_dir = folder_dir / file_dir

        self._ref_joint_angles , self._ref_base_or, self._ref_base_pos, self._ref_com_vels =  [] ,[], [], []


        for i in range(9):
            i = (i+1)*10
            filepaths = [
                f'base_position{i}.npz',
                f'base_orientation{i}.npz',
                f'joint_angles{i}.npz',
                f'com_vels{i}.npz'
            ]
            filepaths = [input_dir / fp for fp in filepaths]

            base_position = self.down_sample_ref(np.load(filepaths[0])['base_position'])
            base_orientation = self.down_sample_ref(np.load(filepaths[1])['base_orientation'])
            joint_angles = self.down_sample_ref(np.load(filepaths[2])['joint_angles'])
            com_vels = self.down_sample_ref(np.load(filepaths[3])['com_vels'])

            self._ref_joint_angles.append(joint_angles)
            self._ref_base_or.append(base_orientation)
            self._ref_base_pos.append(base_position)
            self._ref_com_vels.append(com_vels)

        check = 1

    def load_simulated_data(self):

        self._sim_joint_angles, self._sim_base_or, self._sim_base_pos, self._sim_com_vels = [], [], [], []


        directory = os.path.join(self.path_folder, f'save_data/')

        for i in range(9):
            i=i+1
            filepaths = [

                f'joint_angles_{i}.pt',
                f'com_vels_{i}.pt'
            ]

            filepaths = [directory + fp for fp in filepaths]


            # base_position = self.get_tensor_to_array(torch.load(filepaths[0])['base_position'])
            # base_orientation = self.get_tensor_to_array(torch.load(filepaths[1])['base_orientation'])
            joint_angles = self.get_tensor_to_array(torch.load(filepaths[0]))
            com_vels = self.get_tensor_to_array(torch.load(filepaths[1]))

            self._sim_joint_angles.append(np.vstack((joint_angles, joint_angles[-1, :])))
            # self.sim_base_or.append(base_orientation)
            # self.sim_base_pos.append(base_position)
            self._sim_com_vels.append(np.vstack((com_vels, com_vels[-1, :])))



################################### Ploting function ##########################
    def process_joint_angles(self):

        names = [f'Commanded Velocity {self._desired_vel/10} (m/s)', "Time (s)", "Joint Angles Matching", "Error: Joint Angles Matching", "Time (s)",
                 "Error Joint Angles Matching"]

        joint_angle_index = 10


        self.error_joint_angles = []
        self.errors = []
        self.sim_joint_angles = []
        self.target_joint_angles = []

        for vel in self.velocity_index:

            vel = vel -1

            sim_joint_angles = np.array(self._sim_joint_angles[0])[:, joint_angle_index]
            target_joint_angles = np.array(self._ref_joint_angles[vel])[:, joint_angle_index]


            errors, error_ja = self.RMS_error(sim_joint_angles,target_joint_angles)

            self.sim_joint_angles.append(sim_joint_angles)
            self.target_joint_angles.append(target_joint_angles)
            self.error_joint_angles.append(error_ja)
            self.errors.append(errors)

        if self._plot_single_data:
            self.plot_data(self.target_joint_angles[self._desired_vel-1], self.sim_joint_angles[self._desired_vel-1], names, self.errors[self._desired_vel])
        if self._plot_all_data:
            self.plot_all_data(self.target_joint_angles,self.sim_joint_angles,names)

    def process_velocity(self):

        names = [f'Forward Linear Velocity {self._desired_vel/10} (m/s)', "Time (s)","Velocity (m/s)","Error:Forward Linear Velocity", "Time (s)","Error Velocity (m/s)" ]

        self.error_vel = []
        self.errors = []
        self.sim_vel = []
        self.target_vel = []

        for vel in self.velocity_index:

            vel = vel-1

            sim_vel = np.array(self._sim_com_vels[0])[:,0]
            target_vel = np.array([self._velocity[vel], ] * len(sim_vel))

            errors, error_vel = self.RMS_error(target_vel, sim_vel)

            self.error_vel.append(error_vel)
            self.errors.append(errors)
            self.sim_vel.append(sim_vel)
            self.target_vel.append(target_vel)



        if self._plot_single_data:
            self.plot_data(self.target_vel[self._desired_vel - 1], self.sim_vel[self._desired_vel - 1], names, errors)
        if self._plot_all_data:
            self.plot_all_data(self.target_vel, self.sim_vel,names)


    def RMS_error(self, val1, val2):
        errors = np.square(val1 -  val2)
        error= np.sum(errors)
        error = np.sqrt(error)
        return errors, error

    def percent_error(self, val1, val2):
        errors = val1 - val2
        error = errors/val1
        return errors, error


    def plot_data_vel_pos(self):
        names = [f'Forward Linear Velocity {self._desired_vel / 10} (m/s)', "Time (s)", "Velocity (m/s)",
                 "Joint Positions", "Time (s)", " Joint Positions"]
        target = self.target_vel[self._desired_vel-1]

        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(12, 5))
        axs[0].plot(np.linspace(0, self.max_time, num=len(target)),self.sim_vel[self._desired_vel-1], color='black', linestyle="-",
                    label="Measured")
        axs[0].plot(np.linspace(0, self.max_time, num=len(target)),self.target_vel[self._desired_vel-1] , color='red', linestyle="--",
                    label="Desired")
        axs[0].legend()
        axs[0].set_ylim([-0.5, 1.5])
        axs[0].set_title(names[0])
        axs[0].set_xlabel(names[1])
        axs[0].set_ylabel(names[2])
        axs[1].plot(np.linspace(0, self.max_time, num=len(target)), self.sim_joint_angles[self._desired_vel - 1], color='black', linestyle="-",
                    label="Measured")
        axs[1].plot(np.linspace(0, self.max_time, num=len(target)), self.target_joint_angles[self._desired_vel - 1], color='red', linestyle="--",
                    label="Desired")
        axs[1].legend()
        axs[1].set_ylim([0,3])
        axs[1].set_title(names[3])
        axs[1].set_xlabel(names[4])
        axs[1].set_ylabel(names[5])
        plt.show()


    def plot_data(self,target,sim,names,errors):

        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(12, 5))
        axs[0].plot(np.linspace(0, self.max_time, num=len(target)), sim, color='black', linestyle="-",
                    label="Measured")
        axs[0].plot(np.linspace(0, self.max_time, num=len(target)), target, color='red', linestyle="--",
                    label="Desired")
        axs[0].legend()
        axs[0].set_ylim([-0.5, 1.5])
        axs[0].set_title(names[0])
        axs[0].set_xlabel(names[1])
        axs[0].set_ylabel(names[2])
        axs[1].plot(np.linspace(0, self.max_time, num=len(errors)), errors, color='black', linestyle="-",
                    label="Measured")
        axs[1].plot(np.linspace(0, self.max_time, num=len(errors)), np.zeros(len(errors)), color='red', linestyle="--",
                    label="Desired")
        axs[1].legend()
        axs[1].set_ylim([-0.5,1])
        axs[1].set_title(names[3])
        axs[1].set_xlabel(names[4])
        axs[1].set_ylabel(names[5])
        plt.show()



    def plot_all_data(self,target,sim,names):

        from matplotlib import pyplot as plt
        fig, axs = plt.subplots()
        for i in self.velocity_index:
            i = i-1

            color = tuple(np.random.rand(3))

            axs.plot(np.linspace(0, self.max_time, num=len(sim[i])), sim[i], color=color, linestyle="-")
            axs.plot(np.linspace(0, self.max_time, num=len(target[i])), target[i], color=color, linestyle="--")

        legend = axs.legend(["Measured", "Desired"])
        legend.get_frame().set_edgecolor('black')  # Set legend border color
        legend.get_texts()[0].set_color('black')  # Set color for first legend label
        legend.get_texts()[1].set_color('black')  # Set color for second legend label

        axs.set_title(names[0])
        axs.set_xlabel(names[1])
        axs.set_ylabel(names[2])
        plt.show()

    def get_error_table(self):
        tables = []
        total_error = []

        # Open a file to write the table to.
        with open('output.csv', 'w', newline='') as f:
            # Create a writer object for the file.
            writer = csv.writer(f)


            for vel in self.velocity_index:
                self.sum_error =  self.error_joint_angles[vel] + self.error_vel[vel]


                data = [["joint_angles", self.error_joint_angles[vel]],
                        ["com_vel", self.error_vel[vel]]]



                col_names = ["Parameters", f"v = {self._velocity[vel]} m/s", ]

                total_error.append(self.sum_error)

                # Write the table to the file.
                writer.writerow(col_names)
                writer.writerows(data)

                # Use the `tabulate()` function to create the table.
                table = tabulate(data, headers=col_names)

                # Add the table to the list of tables.
                tables.append(table)

            writer.writerow(["Total Error", total_error])
        # Print the list of tables.

        for table in tables:
            print(table)

    ################################### Utils function ############################
    def get_tensor_to_array(self,tensor):

        tensor_format = tensor[0]
        ind = tensor_format.shape[-1]
        cn = torch.zeros(len(tensor), ind)


        for i in range(len(tensor)):
            cn[i] = tensor[i]
        right_shape = cn.cpu().numpy()

        return right_shape

    def down_sample_ref(self,input):

        down_sample_step= self._sim_joint_angles[0].shape[0]


        step = int(input.shape[0] /down_sample_step)
        down_sampled_input = input[step-1::step]

        return down_sampled_input




if __name__ == "__main__":

    path = 'data/motions/quadruped/mania_pos_rew'
    num_files = 9
    vel = 0.9
    max_time  = 20
    eval = Evaluation(path=path,num_files=num_files,vels=vel, max_time=max_time)
    eval.load_simulated_data()
    eval.load_target_data()
    eval.process_velocity()
    eval.process_joint_angles()
    eval.plot_data_vel_pos()




