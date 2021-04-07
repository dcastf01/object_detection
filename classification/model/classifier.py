import torch 
import torch.nn as nn
from typing import Union

####no implementado####

class Classifier(nn.Module):
    """two class classfication"""

    def __init__(self, backbone, lossfun=RMSELoss(),metrics:Union[None,list]=None):
        super().__init__()
        self.backbone = backbone
        self.lossfun = lossfun
        self.prefix = ""
        self.lossfunc2=CCCLoss()

    def forward(self, image,targets):
        outputs = self.backbone(image)
    
        
        # print(outputs[:,0].shape)
        # print("dffclt",dffclt)
        # print("dscrmn",dscrmn)
        # targets=torch.nn.functional.one_hot(targets)
        loss = self.lossfun(outputs, targets)
     
        metrics = {
            f"{self.prefix}loss_dffclt": loss_dffclt.item(),
            # f"{self.prefix}loss_dscrmn": loss_dscrmn.item(),
          
        }
        # print(metrics)
        return loss, metrics