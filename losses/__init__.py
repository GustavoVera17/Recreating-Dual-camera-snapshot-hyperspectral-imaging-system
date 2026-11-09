"""
Physics-informed loss package for Xie_Rep_2.
"""

from .physics_loss import (
    PhysicsLoss,
    LossResult,
    ClosedLoopPhysicsLoss,
    DualPhysicsLoss,
)

__all__ = [
    'PhysicsLoss',
    'LossResult',
    'ClosedLoopPhysicsLoss',
    'DualPhysicsLoss',
]
