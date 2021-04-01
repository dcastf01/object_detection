import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from classification.loader import Loader
from PIL import Image


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
    
        label=torch.tensor(int(self.data.iloc[index][self.level_to_classifier]))
        label= torch.nn.functional.one_hot(label,num_classes=1716)
        
        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image,label
