
from config import CONFIG
from typing import Union
from classification.model.squeeze_net.torch_squeezeNet import get_squeezenet
from classification.model.vision_transformers.torch_transFG import VisionTransformer
from classification.model.models_with_loss import ModelWithOneLoss

def build_model(model_name:str,loss_fn=None,metrics:Union[None,list]=None):
    #el modelo debe de ser capaz de si le pasas las labels te calcule el resultado de la funcion de perdida
    
    if model_name==CONFIG.ModelName.torch_squeezenet:
        #hace falta añadir la funcion de perdida
        net=get_squeezenet(CONFIG.NUM_CLASSES,loss_fn).to(CONFIG.DEVICE)
        model=ModelWithOneLoss(net)
    elif model_name== CONFIG.ModelName.torch_transFG:
        
        model=None
        
        
        
        
    return model
    
    
