import pandas as pd
import torch
import torch.nn as nn
import copy

from AgentTorch.helpers.general import *

class MakeModule(nn.Module):
    def __init__(self, tensor, tensor_grad):
        super(MakeModule, self).__init__()
        self.param = nn.Parameter(tensor)
        self.param.requires_grad = tensor_grad
        
    def forward(self):
        return self.param
    
    def __call__(self):
        return self.param
    
class Initializer(nn.Module):
    def __init__(self, config, registry):
        super().__init__()
        self.config = config
        self.registry = registry
        
        self.state = {}        
        self.environment, self.agents, self.objects, self.networks = {}, {}, {}, {} #nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), {}
        
        self.fixed_parameters, self.learnable_parameters = {}, {}
        
        self.observation_function, self.policy_function, self.transition_function, self.reward_function = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
                    
    def _initialize_from_default(self, src_val, shape):
        processed_shape = shape
        if type(src_val) == str:
            return src_val
        
        if type(src_val) == list:
            init_value = torch.tensor(src_val)
        else:
            init_value = src_val*torch.ones(size=processed_shape)
        
        return init_value
            
    def _initialize_from_generator(self, initializer_object, initialize_shape, name_root):
        function = initializer_object["generator"]

        params = {}
        for argument in initializer_object["arguments"].keys():
            arg_object = initializer_object["arguments"][argument]
            
            arg_name = f"{name_root}_{argument}"
            
            arg_learnable, arg_shape = arg_object["learnable"], arg_object["shape"]
            arg_init_func = arg_object["initialization_function"]
            
            if arg_init_func is None:
                arg_value = self._initialize_from_default(arg_object["value"], arg_shape)
            else:
                print("!!! dynamic argument are not currently supported. Setup from fixed value !!!")
                return
            
            params[argument] = arg_value
            
            if arg_learnable:
                self.learnable_parameters[arg_name] = arg_value
            else:
                self.fixed_parameters[arg_name] = arg_value
        
        init_value = self.registry.initialization_helpers[function](initialize_shape, params)
        
        return init_value

    def _initialize_property(self, property_object, property_key):
        property_name = property_object["name"]
        property_shape, property_dtype = property_object["shape"], property_object["dtype"]
        property_is_learnable = property_object["learnable"]
        property_initializer = property_object["initialization_function"]
        
        if property_initializer is None:
            property_value = self._initialize_from_default(property_object["value"], property_shape)
        else:
            property_value = self._initialize_from_generator(property_initializer, property_shape, property_key)
                
        if property_is_learnable:
            self.learnable_parameters[property_key] = property_value
        else:
            self.fixed_parameters[property_key] = property_value
                
        return property_value, property_is_learnable
        
    def init_environment(self, key="environment"):
        if self.config["state"][key] is None:
            print("Skipping.. ", key)
            return
            
        for prop in self.config["state"][key].keys():
            property_object = self.config["state"][key][prop]
            property_value, property_is_learnable = self._initialize_property(property_object, property_key=f"{key}_{prop}")
            self.environment[prop] = property_value #MakeModule(property_value, property_is_learnable)
            
#             self.state[key][prop] = self._initialize_property(property_object, property_key=f"{key}_{prop}")
    
    def init_agents(self, key="agents"):
        if self.config["state"][key] is None:
            print("Skipping: ", key)
            return
        
        for instance_type in self.config["state"][key].keys():
            if instance_type == "metadata":
                continue
            
            self.agents[instance_type] = {} #nn.ModuleDict()
            instance_properties = self.config["state"][key][instance_type]["properties"]
            if instance_properties is None:
                continue
            
            for prop in instance_properties.keys():
                property_object = instance_properties[prop]
                property_value, property_is_learnable = self._initialize_property(property_object, property_key=f"{key}_{instance_type}_{prop}")
                self.agents[instance_type][prop] = property_value #MakeModule(property_value, property_is_learnable)
    
    def init_objects(self, key="objects"):
        if self.config["state"][key] is None:
            print("Skipping: ", key)
            return
        
        for instance_type in self.config["state"][key].keys():
            if instance_type == "metadata":
                continue
            
            self.objects[instance_type] = {} #nn.ModuleDict()
            instance_properties = self.config["state"][key][instance_type]["properties"]
            if instance_properties is None:
                continue
            
            for prop in instance_properties.keys():
                property_object = instance_properties[prop]
                property_value, property_is_learnable = self._initialize_property(property_object, property_key=f"{key}_{instance_type}_{prop}")
                self.objects[instance_type][prop] = property_value #MakeModule(property_value, property_is_learnable)
        
    def init_network(self, key="network"):
        if self.config["state"][key] is None:
            print("Skipping.. ", key)
            return
        
        for interaction_type in self.config["state"][key].keys():
            self.networks[interaction_type] = {}
            
            if self.config["state"][key][interaction_type] is None:
                continue
            
            for contact_network in self.config["state"][key][interaction_type].keys():
                self.networks[interaction_type][contact_network] = {}
                
                network_type = self.config["state"][key][interaction_type][contact_network]["type"]
                params = self.config["state"][key][interaction_type][contact_network]["arguments"]
                
                self.networks[interaction_type][contact_network]["graph"], self.networks[interaction_type][contact_network]["adjacency_matrix"] = self.registry.network_helpers[network_type](params)
                
    def simulator(self):
        self.init_environment()        
        self.init_agents(key="agents")
        self.init_objects(key="objects")
        self.init_network()
        
        # track learnable parameters
        self.parameters_dict = nn.ParameterDict(self.learnable_parameters)
        # self.learnable_params_dict = nn.ParameterDict(self.learnable_parameters)

    def _parse_function(self, function_object, name_root):
        generator = function_object["generator"]
        input_variables = function_object["input_variables"]
        output_variables = function_object["output_variables"]
        
        arguments = function_object["arguments"]
        learnable_args, fixed_args = {}, {}
        if arguments is not None:
            for argument in arguments:
                arg_name = f"{name_root}_{argument}"
                
                arg_object = arguments[argument]
                arg_function = arg_object["initialization_function"]
                arg_learnable = arg_object["learnable"]
                arg_shape = arg_object["shape"]
                
                if arg_function is None:
                    arg_value = self._initialize_from_default(arg_object["value"], arg_shape)
                else:
                    arg_value = self._initialize_from_generator(arg_function, arg_shape, name_root=arg_name)
                    
                if arg_learnable:
                    self.learnable_parameters[arg_name] = arg_value
                    learnable_args[argument] = arg_value
                else:
                    self.fixed_parameters[arg_name] = arg_value
                    fixed_args[argument] = arg_value
        
        arguments = {'learnable': learnable_args, 'fixed': fixed_args}
        
        return input_variables, output_variables, arguments

    def substeps(self):
        '''
        define observation, policy and transition functions for each active_agent on each substep
        '''
        
        for substep in self.config["substeps"].keys():
            active_agents = self.config["substeps"][substep]["active_agents"]
            
            self.observation_function[substep], self.policy_function[substep], self.transition_function[substep] = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
            
            for agent_type in active_agents:
                # observation function
                agent_observations = self.config["substeps"][substep]["observation"][agent_type]
                self.observation_function[substep][agent_type] = nn.ModuleDict()
                if agent_observations is not None:
                    for obs_func in agent_observations:
                        input_variables, output_variables, arguments = self._parse_function(agent_observations[obs_func], name_root=f"{agent_type}_observation_{obs_func}")
                        self.observation_function[substep][agent_type][obs_func] = self.registry.observation_helpers[obs_func](self.config, input_variables, output_variables, arguments)
                
                # policy function
                agent_policies = self.config["substeps"][substep]["policy"][agent_type]
                self.policy_function[substep][agent_type] = nn.ModuleDict()
                
                if agent_policies is not None:
                    for policy_func in agent_policies:
                        input_variables, output_variables, arguments = self._parse_function(agent_policies[policy_func], name_root=f"{agent_type}_policy_{policy_func}")
                        self.policy_function[substep][agent_type][policy_func] = self.registry.policy_helpers[policy_func](self.config, input_variables, output_variables, arguments)
                
            # transition function
            substep_transitions = self.config["substeps"][substep]["transition"]
            self.transition_function[substep] = nn.ModuleDict()
            
            for transition_func in substep_transitions:
                input_variables, output_variables, arguments = self._parse_function(substep_transitions[transition_func], name_root=f"_transition_{transition_func}")
                self.transition_function[substep][transition_func] = self.registry.transition_helpers[transition_func](self.config, input_variables, output_variables, arguments)
        
    def initialize(self):
        self.state["current_step"] = 0 #"0" # use string not int for nn.ModuleDict
        self.state["current_substep"] = "0" #'0' # use string not int for nn.ModuleDict
        
        self.simulator()
        self.substeps()
        
        self.state['environment'] = self.environment
        self.state['network'] = self.networks
        self.state['agents'] = self.agents
        self.state['objects'] = self.objects

        self.state['parameters'] = self.parameters_dict
        
    def forward(self):
        self.initialize()

    def __getstate__(self):
        state_dict = self.state.copy()
        state_dict['parameters'] = self.learnable_params_dict.state_dict()
        return state_dict

    def __setstate__(self, state):
        self.learnable_params = nn.ParameterDict(state['parameters'])
        self.state = state
        self.state['parameters'] = self.learnable_params