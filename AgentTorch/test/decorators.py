# test/decorators.py
# tests if the decorators work

# not an actual test, todo: write this using python's unittest
# along with the other tests

import sys
sys.path.insert(0, '../')
from AgentTorch.registry import Registry
from AgentTorch.substep import SubstepAction

@Registry.register_helper("generate_something", "initialization")
def generate_something():
  print("something!")

@Registry.register_helper("AcceptTest", "policy")
class AcceptTest(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        pass

    def forward(self, state, observation):    
        pass

registry = Registry()
print(registry.helpers)
