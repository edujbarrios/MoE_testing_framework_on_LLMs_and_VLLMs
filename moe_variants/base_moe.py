from abc import ABC, abstractmethod
import numpy as np

class BaseMoE(ABC):
    """Base class for all MoE variants."""
    
    def __init__(self, num_experts, input_dim=None):
        self.num_experts = num_experts
        self.input_dim = input_dim
    
    @abstractmethod
    def route(self, inputs):
        """Route inputs to experts."""
        pass
    
    @abstractmethod
    def process(self, inputs):
        """Process inputs using the MoE architecture."""
        pass
    
    def get_routing_info(self):
        """Get information about how inputs are routed."""
        return {
            'num_experts': self.num_experts,
            'routing_type': self.__class__.__name__
        }
