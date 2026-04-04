"""
OpenEnv Email Triage Environment

A complete, real-world OpenEnv environment for training AI agents on email triage tasks.
"""

__version__ = "1.0.0"
__author__ = "Swatimishra8"
__email__ = "swatimis0805@gmail.com"

from env.environment import EmailTriageEnv
from env.models import EmailObservation, EmailAction, EmailReward

__all__ = [
    "EmailTriageEnv",
    "EmailObservation", 
    "EmailAction",
    "EmailReward",
]