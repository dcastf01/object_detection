
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader

from skimage import io

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
        def get_relative_path_img(index):
            
            image_name=self.data.iloc[index]["image_name"]
            make_id=self.data.iloc[index]["make_id"]
            model_id=self.data.iloc[index]["model_id"]
            released_year=self.data.iloc[index]["released_year"]
            extension=self.data.iloc[index]["extension"]
            
            return os.path.join(make_id,model_id,released_year,image_name+"."+extension)
        img_path=os.path.join(self.root_dir_images,get_relative_path_img(index))
        image=io.imread(img_path)
        label=self.data.iloc[index]["model_id"]
        
        if self.transform:
            image=self.transform(image)
        
        
        
        return img_path,label
        
        
    def cut_car(image,x1,y1,x2,y2):
        raise NotImplementedError
        
        
compcar_analisis=CompcarAnalisis(path_csv=config.PATH_COMPCAR_CSV)
compcar_analisis.filter_dataset('viewpoint=="4" or viewpoint=="1"')
print(compcar_analisis.data.head())
loader=CompcarLoader(compcar_analisis.data,root_dir_images=config.PATH_COMPCAR_IMAGES,condition_filter=compcar_analisis.filter)
a=loader[0]
print(a)