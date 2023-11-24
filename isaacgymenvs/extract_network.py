from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
import os
from turtle import forward
import torch
from torch import nn
from torch.nn import Linear,ELU
import time
import yaml



def safe_filesystem_op(func, *args, **kwargs):
    """
    This is to prevent spurious crashes related to saving checkpoints or restoring from checkpoints in a Network
    Filesystem environment (i.e. NGC cloud or SLURM)
    """
    num_attempts = 5
    for attempt in range(num_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(f'Exception {exc} when trying to execute {func} with args:{args} and kwargs:{kwargs}...')
            wait_sec = 2 ** attempt
            print(f'Waiting {wait_sec} before trying again...')
            time.sleep(wait_sec)

    raise RuntimeError(f'Could not execute {func}, give up after {num_attempts} attempts...')

def load_checkpoint(filename):
    print("=> loading checkpoint '{}'".format(filename))
    state = safe_filesystem_op(torch.load, filename)
    return state

def generate_network():

    ## Loading trained model file

    # rel_path = os.path.dirname(os.path.realpath(__file__))
    # file_path = os.path.join(rel_path, '../runs/A1Terrain/nn/A1Terrain.pth')

    #Set home and folder paths
    home_directory = os.path.expanduser('~')
    path = os.path.join(home_directory, 'IsaacGymEnvs/isaacgymenvs')
    rel_path = os.path.dirname(os.path.realpath(__file__))

    #Get checkpoint and cfg paths
    file_name = 'A1_0_0.7_no_dec_base'
    # checkpoint_path = os.path.join(path,f'runs/A1/optimiser/{file_name}.pth')
    checkpoint_path = os.path.join(f'~/PAPER_AMP/saved_checkpoints/vel8/QuadrupedAMP_4000.pth')
    cfg_path = os.path.join(path, 'cfg/task/QuadrupedAMP.yaml')

    #load checkpoint
    trained_model_parameters = load_checkpoint(checkpoint_path)

    #load cfg file
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    normalize = True
    try:
        running_mean = trained_model_parameters["model"]["running_mean_std.running_mean"]
        running_var = trained_model_parameters["model"]["running_mean_std.running_var"]
        running_count = trained_model_parameters["model"]["running_mean_std.count"]
        print("Normalizing parameters found!")
    except:
        running_mean = torch.zeros(48)
        running_var = torch.ones(48)
        running_count = torch.zeros(1)
        normalize = False
        print("No normalization applied!")

    # Initialising a new model
    model = NeuralNet(cfg,running_mean, running_var, running_count, normalize)

    # Initialising model parameters from the trained network
    model.load_state_dict(trained_model_parameters)

    # Example for the network input (required by jit trace)
    example_input = torch.rand(1,48)

    # Acquiring torch script via Tracing
    traced_script_module = torch.jit.trace(model,example_input)

    # Serializing the traced module
    NN_file_name = 'traced_'+file_name+'.jit'
    save_path = os.path.join(rel_path,NN_file_name)

    traced_script_module.save(save_path)

    print("Successfully serialized the network!")
    print("File saved with name" + " " + NN_file_name)


class NeuralNet(nn.Module):
    def __init__(self, cfg, running_mean, running_var, running_count, normalize) -> None:
        super().__init__()

        self.cfg = cfg
        self.normalize = normalize

        if self.normalize:
            self.running_mean_std = RunningMeanStd(torch.tensor(48))
            self.running_mean_std.training = False
            self.running_mean_std.register_buffer("running_mean", running_mean.cpu())
            self.running_mean_std.register_buffer("running_var", running_var.cpu())
            self.running_mean_std.register_buffer("count", running_count.cpu())


        mlp_layers = [
            Linear(48,512),
            ELU(),
            Linear(512,256),
            ELU(),
            Linear(256,128),
            ELU()
        ]

        self.mlp = nn.Sequential(*mlp_layers)

        self.mu = Linear(128,12)
        self.mu_activation = nn.Identity()
        self.sigma = nn.Parameter(torch.zeros(12, requires_grad=True, dtype=torch.float32), requires_grad=True)
        self.sigma_activation = nn.Identity()

    def forward(self,X):

        X = self.preprocess_obs(X)

        if self.normalize:
            X = self.running_mean_std.forward(X)

        mlp_out = self.mlp(X)

        mu_out = self.mu_activation(self.mu(mlp_out))

        action = mu_out
        action = torch.clamp(mu_out,-30,30)

        return action

    def preprocess_obs(self,X):
        """
        The function preprocesses the raw observation inputs before sending them to the neural network
        """

        X = torch.squeeze(X, 0).cpu()

        '''Specific normalisation parameters for A1Terrain'''

        # X[:3] *= self.cfg["env"]["learn"]["linearVelocityScale"]
        # X[3:6] *= self.cfg["env"]["learn"]["angularVelocityScale"]
        # X[6:9] *= self.cfg["env"]["learn"]["projectedGravityScale"]
        # X[9:12] *= torch.tensor(self.cfg["env"]["learn"]["userCommandScale"], requires_grad=False).cpu()
        # X[12:24] *= self.cfg["env"]["learn"]["dofPositionScale"]
        # X[24:36] *= self.cfg["env"]["learn"]["dofVelocityScale"]

        return X

    @torch.jit.ignore
    def load_state_dict(self,checkpoint):
        """
        The function loads the saved actor parameters to a newly initialised model with a matching architecture
        """

        # Pretrained model parameters
        pretrained_dict = checkpoint['model']

        # New model parameters
        model_dict = self.state_dict()

        for key in model_dict.keys():
            if 'a2c_network.' + key in pretrained_dict:
                model_dict[key].copy_(pretrained_dict['a2c_network.' + key])

            elif 'a2c_network.actor_' + key in pretrained_dict:
                model_dict[key].copy_(pretrained_dict['a2c_network.actor_' + key])



if __name__ == "__main__":
    generate_network()
