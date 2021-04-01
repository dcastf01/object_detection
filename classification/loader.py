
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import config
from dataset.compcars.compcar_analisis import CompcarAnalisis
from PIL import Image
# from skimage.viewer import ImageViewer
from torch.utils.data import DataLoader, Dataset
from classification.augmentation import get_transform_from_aladdinpersson
from utils_visualize import plot_examples

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
    
        label=torch.tensor(int(self.data.iloc[index][self.level_to_classifier]))
        label= torch.nn.functional.one_hot(label,num_classes=1716)
        
        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image,label
        
        

def choice_loader_and_splits_dataset(name_dataset:str,BATCH_SIZE:int=16,NUM_WORKERS:int=1) -> dict:
    
    if name_dataset.lower()=="cars196":
        raise NotImplementedError
    elif name_dataset.lower() =="compcars":
        compcar_analisis=CompcarAnalisis(path_csv=config.PATH_COMPCAR_CSV)
        # compcar_analisis.filter_dataset('viewpoint=="4" or viewpoint=="1"')
        # total_count=compcar_analisis.data.shape[0]
       
        train_percent_set=0.7
        valid_percent_set=0.2
        test_percent_set=0.1
        
        # print(compcar_analisis.data.describe())
        # print(compcar_analisis.data.apply(pd.Series.nunique))
        train_ds, validate_ds, test_ds = \
              np.split(compcar_analisis.data.sample(frac=1, random_state=42), 
                       [int(train_percent_set*len(compcar_analisis.data)),
                        int((train_percent_set+valid_percent_set)*len(compcar_analisis.data))])
              
              
        transforms=get_transform_from_aladdinpersson()
        train_transform=transforms["train"]
        val_transform=transforms["val"]
        test_transform=transforms["test"]
        
        train_dataset=CompcarLoader(train_ds,
                         root_dir_images=config.PATH_COMPCAR_IMAGES,
                         transform=train_transform,
                        #  condition_filter=compcar_analisis.filter
                         )
        
        valid_dataset=CompcarLoader(validate_ds,
                         root_dir_images=config.PATH_COMPCAR_IMAGES,
                         transform=val_transform,
                        #  condition_filter=compcar_analisis.filter
                         )
        
        test_dataset=CompcarLoader(test_ds,
                         root_dir_images=config.PATH_COMPCAR_IMAGES,
                         transform=test_transform,
                        #  condition_filter=compcar_analisis.filter
                         )
        
        # train_count = int(0.7 * total_count)
        # valid_count = int(0.2 * total_count)
        # test_count = total_count - train_count - valid_count
        # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        #                             loader,
        #                             (train_count, valid_count, test_count)
        #                             )

        
        
    else:
        raise NotImplementedError

    
    train_dataset_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=NUM_WORKERS
                                                    )
    valid_dataset_loader = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=BATCH_SIZE, 
                    shuffle=True, num_workers=NUM_WORKERS
                                                    )
    test_dataset_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=NUM_WORKERS
                )
    dataloaders = {
        "train": train_dataset_loader,
        "val": valid_dataset_loader,
        "test": test_dataset_loader,
    }
    return dataloaders
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
