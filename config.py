import torch
import os
from enum import Enum
from classification.model.architecture_available import  ArchitectureAvailable
from typing import Union

from dataclasses import dataclass

@dataclass
class CONFIG:
    
    ROOT_WORKSPACE: str=r"D:\programacion\Repositorios\object_detection_TFM"
    ROOT_WORKSPACE: str=""
    #dataset compcar
    PASSWORD_COMPCAR: str="d89551fd190e38"
    PATH_ROOT_COMPCAR: str=os.path.join(ROOT_WORKSPACE,"data","compcars")
    PATH_COMPCAR_CSV: str=os.path.join(PATH_ROOT_COMPCAR,"all_information_compcars.csv")
    PATH_COMPCAR_IMAGES: str=os.path.join(PATH_ROOT_COMPCAR,"image")
    PATH_COMPCAR_LABELS: str=os.path.join(PATH_ROOT_COMPCAR,"label")
    PATH_COMPCAR_TRAIN_REVISITED: str=os.path.join(PATH_ROOT_COMPCAR,"CompCars_revisited_v1.0","bbs_train.txt")
    PATH_COMPCAR_TEST_REVISITED: str=os.path.join(PATH_ROOT_COMPCAR,"CompCars_revisited_v1.0","bbs_test.txt")
    PATH_COMPCAR_MAKE_MODEL_NAME: str=os.path.join(PATH_ROOT_COMPCAR,"misc","make_model_name.mat")
    PATH_COMPCAR_MAKE_MODEL_NAME_CLS: str=os.path.join(PATH_ROOT_COMPCAR,"misc","make_model_names_cls.mat")
    COMPCAR_CONDITION_FILTER: str='viewpoint=="4" or viewpoint=="1"'

    #dataset cars196
    PATH_CARS196_CSV: str=os.path.join(ROOT_WORKSPACE,r"dataset\cars196\all_information_cars196.csv")
    # PATH_CARS196_IMAGES=r"dataset\compcars\all_information_compcars.csv"
    # PATH_CARS196_LABELS=r"dataset\compcars\all_information_compcars.csv"

    #output plots
    PATH_OUTPUT_PLOTS: str=os.path.join(ROOT_WORKSPACE,r"dataset\results")


    #torch config

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # TRAIN_DIR = "data/train"
    # VAL_DIR = "data/val"
    BATCH_SIZE:int = 4
    NUM_CLASSES:int=4455
    LEARNING_RATE:float = 1e-5
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 0
    SEED:int=1
    IMG_SIZE:int=224
    NUM_EPOCHS :int= 10
    LOAD_MODEL :bool= True
    SAVE_MODEL :bool= True

    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"classification/model/checkpoint")
    ARCHITECTURES_AVAILABLE:Enum=ArchitectureAvailable

class ConfigModel(CONFIG):
    
    use_tripletLoss=False
    default_Loss=False
    
    def __init__(self,architecture_name:Union[str,Enum]):#,use_tripletLoss=False,default_Loss=False):
        super(ConfigModel,self).__init__()
        self.architecture_name=self.check_architecture_name_on_list_and_return_architecture_name(architecture_name)
        self.use_tripletLoss=self.use_tripletLoss
        self.default_Loss=self.default_Loss
        
    def check_architecture_name_on_list_and_return_architecture_name(self,architecture_name):
        if architecture_name in self.ARCHITECTURES_AVAILABLE.__members__:
            architecture_name=self.ARCHITECTURES_AVAILABLE[architecture_name]
            return architecture_name
        elif  architecture_name in self.ARCHITECTURES_AVAILABLE:
            
            return architecture_name      
        else:
            print( "you should pick a available model, here you have a list")
            print(self.ARCHITECTURES_AVAILABLE.__members__.keys())
            raise

class ConfigModelTripletLoss(ConfigModel):
    
    def __init__(self,architecture_name:Union[str,Enum]):
        self.use_tripletLoss=True
        
        ConfigModel.__init__(self,architecture_name=architecture_name,)
                                                    # use_tripletLoss=use_tripletLoss)
        
class ConfigModelDefaultLoss(ConfigModel):
    
    def __init__(self,architecture_name:Union[str,Enum]):
        self.default_Loss=True
        ConfigModel.__init__(self,architecture_name=architecture_name,)
                                                    # default_Loss=default_Loss)
    
class ConfigModelTripletAndDefaultLoss(ConfigModelTripletLoss,ConfigModelDefaultLoss):
    def __init__(self,architecture_name:Union[str,Enum]):
        ConfigModelTripletLoss.__init__(self,architecture_name=architecture_name)
        ConfigModelDefaultLoss.__init__(self,architecture_name=architecture_name)
        # super(ConfigModelTripletAndDefaultLoss,self).__init__(architecture_name=architecture_name)
    
a=ConfigModelTripletAndDefaultLoss(architecture_name=ArchitectureAvailable.torch_squeezenet,
                                 
                                    )
print(a)
def get_torch_squeezeNet_tripletloss():
    
    config=ConfigModelTripletLoss(architecture_name=ArchitectureAvailable.torch_squeezenet,
                                 
                                    )