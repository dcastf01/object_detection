
from typing import Union

from classification.model.models_with_loss import (ModelWithOneLoss,
                                                   ModelWithoutLoss,
                                                   ModelWithTripletLoss)
from classification.model.squeeze_net.torch_squeezeNet import get_squeezenet
from classification.model.vision_transformers.torch_transFG import get_transFG
from classification.model.timms_models import ViTBase16,ResNet50
from config import CONFIG


def build_model(
                # architecture_name,
                # loss:bool,
                NUM_CLASSES:int,
                pretrained:bool=True
                ):
    
    # if architecture_name==CONFIG.ARCHITECTURES_AVAILABLE.torch_squeezenet:
    #     model=get_squeezenet(NUM_CLASSES).to(CONFIG.DEVICE)
     
    # elif architecture_name== CONFIG.ARCHITECTURES_AVAILABLE.torch_transFG:
        
    #     model=get_transFG(NUM_CLASS=NUM_CLASSES,
    #                       run_loss_transFG=use_defaultLoss)
        
    #model=ViTBase16(NUM_CLASSES,pretrained=pretrained,transfer_learning=True)
    model=ResNet50(NUM_CLASSES,pretrained=pretrained,transfer_learning=True)
    

    return model
    
    
