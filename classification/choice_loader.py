
import pandas as pd
from timm.data.transforms_factory import transforms_noaug_train
import torch
from config import CONFIG

from torch.utils.data import DataLoader
from classification.augmentation import get_transform_from_aladdinpersson
from classification.loaders.compcars_loader import CompcarLoader,CompcarLoaderTripletLoss
from classification.loaders.cars196_loader import Cars196Loader,Cars196LoaderTripletLoss
import logging
from config import Dataset
from enum import Enum


def choice_loader_and_splits_dataset(name_dataset,
                                     batch_size:int=16,
                                     NUM_WORKERS:int=0,
                                     use_tripletLoss:bool=False) -> dict:
    if isinstance(name_dataset,str):
        name_dataset=Dataset[name_dataset]
    transforms=get_transform_from_aladdinpersson()
    train_transform=transforms["train"]
    val_transform=transforms["val"]
    test_transform=transforms["test"]
        
    if name_dataset==Dataset.cars196:
        NUM_CLASSES=CONFIG.DATASET.CARS196.NUM_CLASSES
        all_ds=pd.read_csv(CONFIG.DATASET.CARS196.PATH_CSV)
        train_ds=all_ds[all_ds["split"]=="train"]
        test_ds=all_ds[all_ds["split"]=="test"]
        
        if use_tripletLoss:
            logging.info("loading cars196 triplet loss")
            loader=Cars196LoaderTripletLoss
            batch_size=batch_size//2
        else:
            loader=Cars196Loader
        
        train_dataset=loader(train_ds,
                        root_dir_images=CONFIG.DATASET.CARS196.PATH_IMAGES,
                        # transform=train_transform,
                        transform=transforms_noaug_train(CONFIG.IMG_SIZE)
                        )
        
        test_dataset=loader(test_ds,
                        root_dir_images=CONFIG.DATASET.CARS196.PATH_IMAGES,
                        # transform=test_transform,                        
                        transform=transforms_noaug_train(CONFIG.IMG_SIZE)
                        )
        
        
    elif name_dataset ==Dataset.compcars:
        def expand_information_useful_on_txt(df:pd.DataFrame)->pd.DataFrame:
            df[["make_id","model_id","released_year","filename"]]=df["Filepath"].str.split("/",expand=True)
            return df
              
        NUM_CLASSES=CONFIG.DATASET.COMPCAR.NUM_CLASSES
        train_ds = pd.read_csv(CONFIG.DATASET.COMPCAR.PATH_TRAIN_REVISITED)
        train_ds=expand_information_useful_on_txt(train_ds)

        test_ds=pd.read_csv(CONFIG.DATASET.COMPCAR.PATH_TEST_REVISITED,)
        test_ds=expand_information_useful_on_txt(test_ds)

        if use_tripletLoss:
            logging.info("loading compcar triplet loss")
            loader=CompcarLoaderTripletLoss
            batch_size=batch_size//2
        else:
            loader=CompcarLoader
        train_dataset=loader(train_ds,
                        root_dir_images=CONFIG.DATASET.COMPCAR.PATH_IMAGES,
                        # transform=train_transform,
                        transform=transforms_noaug_train(CONFIG.IMG_SIZE)
                        )
        
        test_dataset=loader(test_ds,
                        root_dir_images=CONFIG.DATASET.COMPCAR.PATH_IMAGES,
                        # transform=train_transform,
                        transform=transforms_noaug_train (CONFIG.IMG_SIZE)                 
                        )
   
    else:
        raise NotImplementedError
      
    
    train_dataset_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size,
                    shuffle=False, num_workers=CONFIG.NUM_WORKERS,
                    pin_memory=True,
                    drop_last=False,
                    # sampler=
                                                    )

    test_dataset_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size,
                    shuffle=False, num_workers=CONFIG.NUM_WORKERS,
                    pin_memory=True,
                    drop_last=False,
                )
    dataloaders = {
        "train": train_dataset_loader,
        "test": test_dataset_loader,
    }
    return dataloaders,NUM_CLASSES


def test_choice_loader():
    
    dataloaders=choice_loader_and_splits_dataset("compcars",
                                                batch_size=CONFIG.batch_size,
                                                # NUM_WORKERS=CONFIG.NUM_WORKERS
                                                )
    
# test_choice_loader()