"""
pipeline/__init__.py

Re-export key functions from submodules for easy access and
define the public API via __all__.
"""

from .preprocessing import get_preprocessing_func
from .models import get_unet,  get_dog
from .data import patch_generator
from .worker_process import final_process
from .files import get_last_processed_patch_name, clean_merge_neurons_coords_csv_files, check_all_progressions_csv_files_exist

# __all__ controls which names are imported when using:
#   from pipeline import *
__all__ = [
    "get_preprocessing_func",  
    "get_unet",             
    "get_dog",              
    "patch_generator",
    "final_process",
    "get_last_processed_patch_name",
    "clean_merge_neurons_coords_csv_files", 
    "check_all_progressions_csv_files_exist"
]