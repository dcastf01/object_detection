
from config import CONFIG
from typing import Union
from classification.model.squeeze_net.torch_squeezeNet import get_squeezenet
from classification.model.vision_transformers.torch_transFG import VisionTransformer


def build_model(model_name:str,loss_fn=None,metrics:Union[None,list]=None):
    if model_name==CONFIG.ModelName.torch_squeezenet:
        backbone=get_squeezenet(CONFIG.NUM_CLASSES,loss_fn).to(CONFIG.DEVICE)
        
    elif model_name== CONFIG.ModelName.torch_transFG:
        pass
        
        
    model=backbone
        
        
    return model
    
    
