
from config import CONFIG
from typing import Union
from classification.model.squeeze_net.torch_squeezeNet import get_squeezenet
from classification.model.vision_transformers.torch_transFG import get_transFG
from classification.model.models_with_loss import ModelWithOneLoss,ModelWithTripletLoss

def build_model(model_name:str,config_model=None,metrics:Union[None,list]=None):
    #el modelo debe de ser capaz de si le pasas las labels te calcule el resultado de la funcion de perdida
    
    if model_name==CONFIG.ARCHITECTURES_AVAILABLE.torch_squeezenet:
        #hace falta añadir la funcion de perdida
        net=get_squeezenet(CONFIG.NUM_CLASSES).to(CONFIG.DEVICE)
        if config_model.
        model=ModelWithOneLoss(net)
    elif model_name== CONFIG.ARCHITECTURES_AVAILABLE.torch_transFG:
        
        model=get_transFG(NUM_CLASS=CONFIG.NUM_CLASSES)
        
    model=ModelWithTripletLoss(model)
    return model
    
    
