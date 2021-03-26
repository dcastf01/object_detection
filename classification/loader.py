
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import config
from dataset.compcars.compcar_analisis import CompcarAnalisis
from PIL import Image
# from skimage.viewer import ImageViewer
from torch.utils.data import DataLoader, Dataset

from classification.augmentation import get_transform_from_aladdinpersson
from utils import plot_examples

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
                  transform=None,condition_filter:str=None,level_to_classifier:str="model_id"):
        self.level_to_classifier=level_to_classifier
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
        
        def cut_car(image,index):
            x1=self.data.iloc[index]["x1"]
            x2=self.data.iloc[index]["x2"]
            y1=self.data.iloc[index]["y1"]
            y2=self.data.iloc[index]["y2"]
            w=x2-x1
            h=y2-y1
            
            return transforms.functional.crop(image,y1,x1,h,w)
    
        img_path=os.path.join(self.root_dir_images,get_relative_path_img(index))
        image_global=Image.open(img_path).convert("RGB")
        image=np.array(cut_car(image_global,index))
      
        label=self.data.iloc[index][self.level_to_classifier]
        
        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image,label
        
        
            
def test():
    
    compcar_analisis=CompcarAnalisis(path_csv=config.PATH_COMPCAR_CSV)
    compcar_analisis.filter_dataset('viewpoint=="4" or viewpoint=="1"')
    print(compcar_analisis.data.head())
    transform=get_transform_from_aladdinpersson()
    loader=CompcarLoader(compcar_analisis.data,
                         root_dir_images=config.PATH_COMPCAR_IMAGES,
                         transform=transform,
                         condition_filter=compcar_analisis.filter)
    images=[]
    for i in range(15):
        image,label=loader[0]
        images.append(image.permute(1, 2, 0))
    
    plot_examples(images)
    
# test()
