import torch.nn as nn

class ModelWithOneLoss(nn.Module):
    
    def __init__(self,net,loss_fn=nn.CrossEntropyLoss()):
        super(ModelWithOneLoss,self).__init__()
        self.net=net
        self.loss_fn=loss_fn
        
    def forward(self,x,labels=None):
        
        preds=self.net(x)
        
        if labels is not None:
            
            loss=self.loss_fn(preds,labels)
            
            return loss,preds
        else:
 
            return preds