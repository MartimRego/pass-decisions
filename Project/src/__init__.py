"""
Pass Decision Analysis - MDP Framework
=======================================

A modular framework for analyzing pass decision-making in soccer using
Markov Decision Processes, based on Van Roy et al. (2020) methodology.

Modules:
--------
- data_processing: Load and classify pass events
- state_action: State space and action space encoding
- mdp: MDP construction (transitions, policy, rewards)
- success_modeling: Quality-quantity trade-off modeling
- analysis: Fundamental matrix, counterfactual analysis
- visualization: Plotting and heat maps
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import data_processing
from . import state_action
from . import mdp
from . import success_modeling
from . import analysis
from . import visualization

__all__ = [
    'data_processing',
    'state_action',
    'mdp',
    'success_modeling',
    'analysis',
    'visualization'
]
