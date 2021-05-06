import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from classification.augmentation import get_transform_from_aladdinpersson
from classification.loaders.loader import Loader
from config import CONFIG
from dataset.compcars.compcar_analisis import CompcarAnalisis
from PIL import Image
from utils_visualize import plot_examples


class CompcarLoaderBasic(Loader):
    def __init__(self, df:pd.DataFrame,root_dir_images:str,
                  transform=None,condition_filter:str=None):
        df=self.generate_label_id_on_df(df)
        super().__init__(df=df,root_dir_images=root_dir_images,
                         transform=transform,condition_filter=condition_filter,
                         )
        
    def generate_label_id_on_df(self,df:pd.DataFrame)->dict:
            df['id'] = df.groupby(["make_id",'model_id','released_year']).ngroup()
        
            return df
    def _get_image_and_label(self,index):    
    
        def get_relative_path_img(index):

            return self.data.iloc[index]["Filepath"]
        def cut_car(image,index):
            X=self.data.iloc[index]["X"]
            Y=self.data.iloc[index]["Y"]

            w=self.data.iloc[index]["Width"]
            h=self.data.iloc[index]["Height"]
            
            return transforms.functional.crop(image,Y,X,h,w)
    
        img_path=os.path.join(self.root_dir_images,get_relative_path_img(index))
        filename=img_path.split("/")[-1]
   
        image_global=Image.open(img_path).convert("RGB")
        if self.need_crop:
            image_cut=(cut_car(image_global,index))
        else:
            image_cut=image_global
        # image=np.array(image_cut)
        image=image_cut
        label=torch.tensor(int(self.data.iloc[index]["id"]))
        
        if self.transform:
            image=self.transform(image)
            # augmentations = self.transform(image=image)
            # image = augmentations["image"]

        return image,label,filename

    def __getitem__ (self,index):
        NotImplementedError
        
class CompcarLoader(CompcarLoaderBasic):
    def __init__(self, df:pd.DataFrame,root_dir_images:str,
                  transform=None,condition_filter:str=None,
                  ):
        super(CompcarLoader,self).__init__(df,root_dir_images,
                                            transform,condition_filter,
                                          )
        self.need_crop=False #esto es solo para debuguear, no olvidar quitar
    
    def __getitem__ (self,index):
        image,label,filename=self._get_image_and_label(index)
        return image,label,filename
    
class CompcarLoaderTripletLoss(CompcarLoaderBasic):
    def __init__(self, df:pd.DataFrame,root_dir_images:str,
                  transform=None,condition_filter:str=None,):
        
        super(CompcarLoaderTripletLoss,self).__init__(df,root_dir_images,
                                            transform,condition_filter,
                                            )
                
        self.labels=(self.data.id.to_numpy())
        self.labels_set = set(self.data.id.to_numpy())
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                     for label in self.labels_set}
        self.index=list(self.data.index.values)
        
    def __getitem__(self,index):
        
        anchor_img,anchor_label=self._get_image_and_label(index)
        #https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
        #https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
        

        positive_index=index
        i=0
        while positive_index == index:
            
            # positive_list=self.index[self.index!=index][self.labels[self.index!=index]==anchor_label]
            positive_index = np.random.choice(self.label_to_indices[anchor_label.item()])
            i+=1
            if i>4:
                positive_index=index
                break
        
        negative_label=np.random.choice(list(self.labels_set-set([anchor_label.item()])))
        negative_index=np.random.choice(self.label_to_indices[negative_label])            
        # 
        # positive_index = random.choice(positive_list)
        
        positive_img,positive_label=self._get_image_and_label(positive_index)
        negative_img,negative_label=self._get_image_and_label(negative_index)
        
        return (anchor_img, positive_img, negative_img),(anchor_label,positive_label,negative_label)
        

def test_CompcarLoader():
    
    compcar_analisis=CompcarAnalisis(path_csv=CONFIG.DATASET.COMPCAR.PATH_CSV)
    # compcar_analisis.filter_dataset('viewpoint=="4" or viewpoint=="1"')
    print(compcar_analisis.data.head())
    transform_train=get_transform_from_aladdinpersson()["train"]
    loader=CompcarLoader(compcar_analisis.data,
                         root_dir_images=CONFIG.DATASET.COMPCAR.PATH_IMAGES,
                         transform=transform_train,
                         condition_filter=compcar_analisis.filter)
    images=[]
    for i in range(15):
        image,label=loader[0]
        images.append(image.permute(1, 2, 0))
    
    plot_examples(images)
    
# test_CompcarLoader()
