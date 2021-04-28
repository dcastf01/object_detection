import torch.nn as nn
import timm
import torch
from config import CONFIG
class ViTBase16(nn.Module):
    def __init__(self, num_classes, pretrained=False,transfer_learning=True):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_224_in21k", pretrained=pretrained,
                                      
                                       )

        if transfer_learning:
            for param in self.model.parameters():
                param.requires_grad=False

        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        

    def forward(self, x):
        x = self.model(x)
        return x

