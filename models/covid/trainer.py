'''Command: python trainer.py --c config_opt_llm.yaml'''
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.optim as optim
import sys
import torch.nn as nn

AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'

sys.path.insert(0, AGENT_TORCH_PATH)

from simulator import get_registry, get_runner
from AgentTorch.helpers import read_config
from calibnn import CalibNN, LearnableParams

# *************************************************************************
# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="AgentTorch: million-scale, differentiable agent-based models"
)
parser.add_argument(
    "-c", "--config", default="config.yaml", help="Name of the yaml config file with the parameters."
)
# *************************************************************************

args = parser.parse_args()
config_file = args.config
print("Running experiment with config file: ", config_file)

CALIB_MODE = 'learnable_param' # i -> internal_param; external_param -> nn.Parameter; learnable_param -> learnable_parameters; nn -> CalibNN

config = read_config(config_file)
registry = get_registry()
runner = get_runner(config, registry)

device = torch.device(runner.config['simulation_metadata']['device'])
num_episodes = runner.config['simulation_metadata']['num_episodes']
num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']

class LearnableParamsWrapper(nn.Module):
    def __init__(self, num_params, device, scale_output='abm-covid'):
        super().__init__()
        self.learnable_params_model = LearnableParams(num_params, device, scale_output)
        self.learnable_params_output = None

    def forward(self):
        self.learnable_params_output = self.learnable_params_model()
        return self.learnable_params_output

    def parameters(self):
        return self.learnable_params_model.parameters

runner.init()

named_params_learnable = [(name, param) for (name, param) in runner.named_parameters() if param.requires_grad]
print("named learnable_params: ", named_params_learnable)

learning_rate = runner.config['simulation_metadata']['learning_params']['lr']
betas = runner.config['simulation_metadata']['learning_params']['betas']

if CALIB_MODE == 'internal_param':
    learnable_params = [param for param in runner.parameters() if param.requires_grad]
    opt = optim.Adam(learnable_params, lr=learning_rate,betas=betas)
elif CALIB_MODE == 'external_param':
    R = nn.Parameter(torch.tensor([4.10]))
    opt = optim.Adam([R], lr=learning_rate,betas=betas)
    runner.initializer.transition_function['0']['new_transmission'].learnable_args.R2 = R

elif CALIB_MODE == 'learnable_param':
    learn_model = LearnableParams(num_params=1, device=device)
    opt = optim.Adam(learn_model.parameters(), lr=learning_rate, betas=betas)

def _get_parameters(CALIB_MODE):
    if CALIB_MODE == 'learnable_param':
        new_R = learn_model()
        print("R shape: ", new_R.shape)
        return new_R

def _set_parameters(new_R):
    '''Only sets R value for now..'''
    breakpoint()
    runner.initializer.transition_function['0']['new_transmission'].external_R = new_R
    #runner.initializer.transition_function['0']['new_transmission'].learnable_args.R2 = new_R

for episode in range(num_episodes):
    opt.zero_grad()
    if CALIB_MODE != 'internal_param' and CALIB_MODE != 'external_param':
        print("Calib Mode: ", CALIB_MODE)
        new_R = _get_parameters(CALIB_MODE)
        print("new R: ", new_R)
        _set_parameters(new_R)

    runner.step(num_steps_per_episode)
    
    traj = runner.state_trajectory[-1][-1]
    daily_infections_arr = traj['environment']['daily_infected'] #()
    
    loss_val = daily_infections_arr.sum() # test loss for now. will be replaced after
    loss_val.backward()

    # Check the gradients for all parameters in the optimizer
    for param_group in opt.param_groups:
        for param in param_group['params']:
            print(f"Parameter: {param.data}, Gradient: {param.grad}")

    breakpoint()

    opt.step()

    runner.reset()