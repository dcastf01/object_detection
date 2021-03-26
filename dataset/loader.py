
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader


from dataset import config

from dataset.compcars.compcar_analisis import CompcarAnalisis

class Loader (Dataset):
    
    def __init__ (self, df:pd.DataFrame,root_dir_images:str,
                  transform=None,condition_filter:str=None):
        self.data=df
        self.root_dir_images=root_dir_images
        
        self.transform=transform
        if condition_filter is not None:
            self.condition_filter=condition_filter
            self.data=self.data.query(condition_filter)
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,index):
        
        raise NotImplementedError

class CompcarLoader(Loader):
    def __init__(self, df:pd.DataFrame,root_dir_images:str,
                  transform=None,condition_filter:str=None):
        
        super().__init__(df=df,root_dir_images=root_dir_images,
                         transform=transform,condition_filter=condition_filter)
        
    def __getitem__ (self,index):
        pass
        
        
        
compcar_analisis=CompcarAnalisis(path_csv=config.PATH_COMPCAR_CSV)
compcar_analisis.filter_dataset('viewpoint=="4" or viewpoint=="1"')

loader=CompcarLoader(compcar_analisis.data,root_dir_images=config.PATH_COMPCAR_IMAGES,condition_filter=compcar_analisis.filter)
a=loader[0]
print(a)