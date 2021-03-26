
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd

from torch.utils.data import Dataset,DataLoader


class Loader (Dataset):
    
    def __init__ (self, csv_file:str,root_dir:str,
                  transform=None,filter:str=None):
        pass