from timm.models import resnet
from timm.models.helpers import build_model_with_cfg
import torch.nn as nn
import timm
from config import CONFIG
from enum import Enum

    
import logging
class TimmModel(nn.Module):

    def __init__(self,model_name, pretrained=False,transfer_learning=True):

        super(TimmModel, self).__init__()
        logging.info(f"creating the model {model_name} ")

        self.model = timm.create_model( 
                                       model_name,
                                       pretrained=pretrained,
                                      
                                       )

        if transfer_learning:
            for param in self.model.parameters():
                param.requires_grad=False

        
        
    def forward(self, x):
        x = self.model(x)
        return x
    

    
class ViTBase16(TimmModel):
    def __init__(self, num_classes, pretrained=False,transfer_learning=True):
   
        model_name="vit_base_patch16_224_in21k"
        
        
        super(ViTBase16, self).__init__(model_name,
                                        pretrained,
                                        transfer_learning)
        
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        
  
class ResNet50(TimmModel):
    def __init__(self, num_classes, pretrained=False,transfer_learning=True):

        model_name="resnet50"
        # cfg_modified=timm.models.resnet.default_cfgs[model_name]
        # cfg_modified["input_size"]=(3,600,600)
        # model_args = dict(block=timm.models.resnet.Bottleneck, layers=[3, 4, 6, 3])
        # model=build_model_with_cfg(resnet.ResNet,
        #                      model_name,
        #                      pretrained=pretrained,
        #                      default_cfg=cfg_modified,
        #                      **model_args
        #                      )
        super(ResNet50, self).__init__(model_name,
                                            pretrained,
                                            transfer_learning)   
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    # def forward(self, x):
    #     x = self.model(x)
    #     return x