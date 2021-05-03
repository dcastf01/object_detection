from enum import Enum
from typing import Union

# from config import CONFIG

from classification.model.architecture_available import 

class ExperimentNames(Enum):
    SqueezeNet="torch_SqueezeNet_DefaultLoss"
    TimmVIT="tim_vit"
    TimmResNet50="tim_resnet50"
    
class LossNames(Enum):
    Crossentropy="crossentropy"
    TripletLoss="triplet loss"

class ExperimentConfig():
    def __init__(self,experiment_name:Enum,loss:Enum=LossNames.Crossentropy,pretrained:bool=True) -> None:
        self.ExperimentName=experiment_name
        self.Loss=loss
        self.pretrained=pretrained
    

default_experiments={
    
    "vit": ExperimentConfig(ExperimentNames.TimmVIT,
                            LossNames.Crossentropy,
                            
                            )
    
    
}