import os
from config import CONFIG
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from classification.loaders.loader import Loader
from PIL import Image
from utils_visualize import plot_examples

from dataset.compcars.compcar_analisis import CompcarAnalisis
from classification.augmentation import get_transform_from_aladdinpersson

class CompcarLoader(Loader):
    def __init__(self, df:pd.DataFrame,root_dir_images:str,
                  transform=None,condition_filter:str=None):
        df=self.generate_label_id_on_df(df)
        super().__init__(df=df,root_dir_images=root_dir_images,
                         transform=transform,condition_filter=condition_filter)
        
    def generate_label_id_on_df(self,df:pd.DataFrame)->dict:
            df['id'] = df.groupby(["make_id",'model_id','released_year']).ngroup()
        
            return df
        
    def __getitem__ (self,index):
        def get_relative_path_img(index):
            
            # image_name=self.data.iloc[index]["image_name"]
            # make_id=self.data.iloc[index]["make_id"]
            # model_id=self.data.iloc[index]["model_id"]
            # released_year=self.data.iloc[index]["released_year"]
            # extension=self.data.iloc[index]["extension"]
            
            # return os.path.join(make_id,model_id,released_year,image_name+"."+extension)
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


def test_CompcarLoader():
    
    compcar_analisis=CompcarAnalisis(path_csv=config.PATH_COMPCAR_CSV)
    # compcar_analisis.filter_dataset('viewpoint=="4" or viewpoint=="1"')
    print(compcar_analisis.data.head())
    transform_train=get_transform_from_aladdinpersson()["train"]
    loader=CompcarLoader(compcar_analisis.data,
                         root_dir_images=config.PATH_COMPCAR_IMAGES,
                         transform=transform_train,
                         condition_filter=compcar_analisis.filter)
    images=[]
    for i in range(15):
        image,label=loader[0]
        images.append(image.permute(1, 2, 0))
    
    plot_examples(images)
    
# test_CompcarLoader()