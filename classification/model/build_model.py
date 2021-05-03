
from typing import Union

from classification.model.timms_models import ViTBase16,ResNet50
from config import CONFIG
from enum import Enum
class Models_available(Enum):
    ViTBase16="vit_base_patch16_224_in21k"
    ResNet50="resnet50"

def build_model(
                model_selected,
                # loss:bool,
                NUM_CLASSES:int,
                pretrained:bool=True
                ):
    


    if model_selected==Models_available.ResNet50:
        model=ResNet50(NUM_CLASSES,pretrained=pretrained,transfer_learning=True) 
    elif model_selected==Models_available.ViTBase16:
        model=ViTBase16(NUM_CLASSES,pretrained=pretrained,transfer_learning=True)
    
    

    return model
    
    
