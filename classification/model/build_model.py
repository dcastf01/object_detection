
from typing import Union

from classification.model.timms_models import ViTBase16,ResNet50
from config import ModelsAvailable

def build_model(
                model_selected,
                # loss:bool,
                NUM_CLASSES:int,
                pretrained:bool=True,
                transfer_learning:bool=True
                
                ):
    


    if model_selected==ModelsAvailable.ResNet50:
        model=ResNet50(NUM_CLASSES,pretrained=pretrained,transfer_learning=True) 
    elif model_selected==ModelsAvailable.ViTBase16:
        model=ViTBase16(NUM_CLASSES,pretrained=pretrained,transfer_learning=True)
    
    

    return model
    
    
