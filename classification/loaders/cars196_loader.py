import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from classification.augmentation import get_transform_from_aladdinpersson
from classification.loaders.loader import Loader
from config import CONFIG
from PIL import Image
from utils_visualize import plot_examples


class Cars196LoaderBasic(Loader):
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
        image_global=Image.open(img_path).convert("RGB")
        image=np.array(cut_car(image_global,index))
    
        label=torch.tensor(int(self.data.iloc[index]["id"]))
        # label= torch.nn.functional.one_hot(label,num_classes=config.NUM_CLASSES)
        
        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image,label

    def __getitem__ (self,index):
        NotImplementedError
        
class Cars196Loader(Cars196LoaderBasic):
    def __init__(self, df:pd.DataFrame,root_dir_images:str,
                  transform=None,condition_filter:str=None,
                  ):
        super(Cars196Loader,self).__init__(df,root_dir_images,
                                            transform,condition_filter,
                                          )
    
    def __getitem__ (self,index):
        image,label=self._get_image_and_label(index)
        return image,label
    
class Cars196LoaderTripletLoss(Cars196LoaderBasic):
    def __init__(self, df:pd.DataFrame,root_dir_images:str,
                  transform=None,condition_filter:str=None,):
        
        super(Cars196LoaderTripletLoss,self).__init__(df,root_dir_images,
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
        while positive_index == index:
            # positive_list=self.index[self.index!=index][self.labels[self.index!=index]==anchor_label]
            positive_index = np.random.choice(self.label_to_indices[anchor_label.item()])
        
        negative_label=np.random.choice(list(self.labels_set-set([anchor_label.item()])))
        negative_index=np.random.choice(self.label_to_indices[negative_label])            
        # 
        # positive_index = random.choice(positive_list)
        
        positive_img,positive_label=self._get_image_and_label(positive_index)
        negative_img,negative_label=self._get_image_and_label(negative_index)
        
        return (anchor_img, positive_img, negative_img),(anchor_label,positive_label,negative_label)
        

def test_Cars196Loader():
    
    cars196_analisis=pd.read_csv(CONFIG.DATASET.CARS196.PATH_CSV)
    # cars196_analisis.filter_dataset('viewpoint=="4" or viewpoint=="1"')
    print(cars196_analisis.head())
    transform_train=get_transform_from_aladdinpersson()["train"]
    loader=Cars196Loader(cars196_analisis,
                         root_dir_images=CONFIG.DATASET.CARS196.PATH_IMAGES,
                         transform=transform_train,
                         condition_filter=cars196_analisis.filter)
    images=[]
    for i in range(15):
        image,label=loader[0]
        images.append(image.permute(1, 2, 0))
    
    plot_examples(images)
    
test_Cars196Loader()
