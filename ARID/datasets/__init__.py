# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu


from .ARID import ARID
from .ARID_mod import ARID_MOD
from .ucf101 import ucf101
from .pipe_datasets import Action_DATASETS

__all__ = ('ucf101', 'ARID', 'Action_DATASETS')
