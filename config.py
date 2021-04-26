import torch
import os
from enum import Enum
from classification.model.architecture_available import  ArchitectureAvailable
from typing import Union

from dataclasses import dataclass

import ml_collections
ROOT_WORKSPACE: str=r"D:\programacion\Repositorios\object_detection_TFM"
ROOT_WORKSPACE: str=""
@dataclass
class CONFIG:
    
    
    #torch config
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # TRAIN_DIR = "data/train"
    # VAL_DIR = "data/val"
    BATCH_SIZE:int = 128
    NUM_CLASSES:int=4455
    LEARNING_RATE:float = 1e-3
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 4
    SEED:int=1
    IMG_SIZE:int=224
    NUM_EPOCHS :int= 50
    LOAD_MODEL :bool= True
    SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"classification/model/checkpoint")
    
    ARCHITECTURES_AVAILABLE:Enum=ArchitectureAvailable
    # def _get_json(self):
    #     import json
    #     return json.dumps(self.__dict__)
    class DATASET:
        
        
        class COMPCAR:
            
            #dataset compcar
            PASSWORD_ZIP: str="d89551fd190e38"
            PATH_ROOT: str=os.path.join(ROOT_WORKSPACE,"data","compcars")
            PATH_CSV: str=os.path.join(PATH_ROOT,"all_information_compcars.csv")
            PATH_IMAGES: str=os.path.join(PATH_ROOT,"image")
            PATH_LABELS: str=os.path.join(PATH_ROOT,"label")
            PATH_TRAIN_REVISITED: str=os.path.join(PATH_ROOT,"CompCars_revisited_v1.0","bbs_train.txt")
            PATH_TEST_REVISITED: str=os.path.join(PATH_ROOT,"CompCars_revisited_v1.0","bbs_test.txt")
            PATH_MAKE_MODEL_NAME: str=os.path.join(PATH_ROOT,"misc","make_model_name.mat")
            PATH_MAKE_MODEL_NAME_CLS: str=os.path.join(PATH_ROOT,"misc","make_model_names_cls.mat")
            COMPCAR_CONDITION_FILTER: str='viewpoint=="4" or viewpoint=="1"'
        class CARS196:
            #dataset cars196
            PATH_ROOT:str= os.path.join(ROOT_WORKSPACE,"data","cars196")
            PATH_CARS196_CSV: str=r"dataset\cars196\all_information_cars196.csv"
            PATH_IMAGES:str=os.path.join(PATH_ROOT,)
            PATH_LABELS:str=os.path.join(PATH_ROOT,)
            # PATH_CARS196_IMAGES=r"dataset\compcars\all_information_compcars.csv"
            # PATH_CARS196_LABELS=r"dataset\compcars\all_information_compcars.csv"

        #output plots
        PATH_OUTPUT_PLOTS: str=os.path.join("dataset","results")
    

class WrapperConfigModel(CONFIG):
    use_tripletLoss=False
    default_Loss=False
    architecture_name=None

class ConfigArchitecture(WrapperConfigModel):
    
    def __init__(self,architecture_name:Union[str,Enum]):
        # super(ConfigArchitecture,self).__init__()
        self.architecture_name=self.check_architecture_name_on_list_and_return_architecture_name(architecture_name)
            
    def check_architecture_name_on_list_and_return_architecture_name(self,architecture_name):
        if architecture_name in self.ARCHITECTURES_AVAILABLE.__members__:
            architecture_name=self.ARCHITECTURES_AVAILABLE[architecture_name]
            return architecture_name
        elif  architecture_name in self.ARCHITECTURES_AVAILABLE:
            
            return architecture_name      
        else:
            print( "you should pick a available model, here you have a list")
            print(self.ARCHITECTURES_AVAILABLE.__members__.keys())
            raise "you should pick a available model, here you have a list"


class ConfigModelDefaultLoss(WrapperConfigModel):
    
    def __init__(self):
        self.default_Loss=True
        # ConfigModel.__init__(self,architecture_name=architecture_name,)
                                                    # default_Loss=default_Loss)
                                                    
class ConfigModelTripletLoss(WrapperConfigModel):
    
    def __init__(self):
        self.use_tripletLoss=True
        
    
class ConfigModelDefaultLossAndTripletLoss(ConfigModelTripletLoss,ConfigModelDefaultLoss):
    def __init__(self):
        ConfigModelTripletLoss.__init__(self)
        ConfigModelDefaultLoss.__init__(self)
        # super(ConfigModelTripletAndDefaultLoss,self).__init__(architecture_name=architecture_name)
    
class TorchSqueezenet(ConfigArchitecture ):
    def __init__(self):
        architecture_name=ArchitectureAvailable.torch_squeezenet
        ConfigArchitecture.__init__(self,architecture_name)
        
class TorchtransFG(ConfigArchitecture ):
    def __init__(self):
        architecture_name=ArchitectureAvailable.torch_transFG
        ConfigArchitecture.__init__(self,architecture_name)

class TorchSqueeNetDefaultLossLoss(TorchSqueezenet,ConfigModelDefaultLoss):
    def __init__(self):
        TorchSqueezenet.__init__(self)
        ConfigModelDefaultLoss.__init__(self)

class TorchSqueeNetTripletLoss(TorchSqueezenet,ConfigModelTripletLoss):
    def __init__(self):
        TorchSqueezenet.__init__(self)
        ConfigModelTripletLoss.__init__(self)
        
class TorchSqueeNetDefaultLossAndTripletLoss(TorchSqueezenet,ConfigModelDefaultLossAndTripletLoss):
    def __init__(self):
        TorchSqueezenet.__init__(self)
        ConfigModelDefaultLossAndTripletLoss.__init__(self)

class TorchtransFGDefaultLossLoss(TorchtransFG,ConfigModelDefaultLoss):
    def __init__(self):
        TorchtransFG.__init__(self)
        ConfigModelDefaultLoss.__init__(self)

class TorchtransFGTripletLoss(TorchtransFG,ConfigModelTripletLoss):
    def __init__(self):
        TorchtransFG.__init__(self)
        ConfigModelTripletLoss.__init__(self)
        
class TorchtransFGDefaultLossAndTripletLoss(TorchtransFG,ConfigModelDefaultLossAndTripletLoss):
    def __init__(self):
        TorchtransFG.__init__(self)
        ConfigModelDefaultLossAndTripletLoss.__init__(self)
        

