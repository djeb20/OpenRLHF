"""
Module for defining constraints in reinforcement learning.
"""

from typing import Optional
from torch import nn

class BaseConstraint(nn.Module):
    """Abstract base class for constraints."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
class CommutativityConstraint(BaseConstraint):
    """Example commutativity constraint."""
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        return 0 # Placeholder implementation

class ConstraintLoss(nn.Module):
    """
    Class for collecting constraint losses and combining them with policy loss.
    """

    def __init__(
        self, 
        constraints: list = None,
        constraint_weights: Optional[list] = None,
    ):
        super().__init__()

        # Initialize the list of constraint loss functions
        self.constraint_loss_fns = []

        if 'commutativity' in constraints:
            self.constraint_loss_fns.append(CommutativityConstraint())

        # Assume uniform weights if none provided
        if constraint_weights is None:
            self.constraint_weights = [1.0] * len(self.constraint_loss_fns)
        else:
            self.constraint_weights = constraint_weights

    def forward(
        self, 
        *args,
        **kwargs,
        ):

        loss = 0

        for constraint_fn, weight in zip(self.constraint_loss_fns, self.constraint_weights):
            loss += weight * constraint_fn(*args, **kwargs)

        return loss