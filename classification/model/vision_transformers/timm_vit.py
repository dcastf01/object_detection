import torch.nn as nn
import timm
import torch
from config import CONFIG
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False,trained:bool=False):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True
                                      
                                       )
        if trained:
            self.model.load_state_dict(torch.load(MODEL_PATH))

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x

m=ViTBase16(10)   
o=m(torch.randn(2,3,224,224))
print(f'Original shape: {o.shape}')
o = m.forward_features(torch.randn(2, 3, 299, 299))
print(f'Unpooled shape: {o.shape}')
print("Available Vision Transformer Models: ")
print(timm.list_models("vit*"))