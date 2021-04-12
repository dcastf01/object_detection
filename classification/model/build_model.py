
from config import CONFIG
from typing import Union
from classification.model.squeeze_net.torch_squeezeNet import get_squeezenet
from classification.model.vision_transformers.torch_transFG import get_transFG
from classification.model.models_with_loss import ModelWithOneLoss,ModelWithTripletLoss

def build_model(architecture_name,use_defaultLoss:bool,
                use_tripletLoss:bool,
                metrics:Union[None,list]=None):
    
    if architecture_name==CONFIG.ARCHITECTURES_AVAILABLE.torch_squeezenet:
        model=get_squeezenet(CONFIG.NUM_CLASSES).to(CONFIG.DEVICE)
        if use_defaultLoss:
            model=ModelWithOneLoss(model)
    elif architecture_name== CONFIG.ARCHITECTURES_AVAILABLE.torch_transFG:
        
        model=get_transFG(NUM_CLASS=CONFIG.NUM_CLASSES,
                          run_loss_transFG=use_defaultLoss)
    
    if use_tripletLoss:
        model=ModelWithTripletLoss(model)
    return model
    
    
