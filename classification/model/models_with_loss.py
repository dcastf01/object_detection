import torch.nn as nn
import torch
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
        
        
class ModelWithTripletLoss(nn.Module):
    
    def __init__(self,model):
        super(ModelWithTripletLoss,self).__init__()
        
        self.model=model
        self.triplet_loss=nn.TripletMarginLoss()
        
    def forward(self,x,labels=None):
        if len(x)==3:
            anchor_imgs,positive_imgs,negative_imgs=x
            anchor_labels,positive_labels,negative_labels=labels
            
            if labels is not None:
                
                anchor_loss,anchor_preds=self.model(anchor_imgs,anchor_labels)
                positive_loss,positive_preds=self.model(positive_imgs,positive_labels)
                negative_loss,negative_preds=self.model(negative_imgs,negative_labels)
                
                loss_crosentropy=torch.sum(anchor_loss,positive_loss,negative_loss)
                loss_triplet=self.triplet_loss(anchor_preds,positive_loss,negative_loss)
                
                return loss,anchor_preds
            else:
                anchor_preds=self.model(anchor)
                return anchor_preds
        else:
            anchor=x
            if labels is not None:
                anchor_loss,anchor_preds=self.model(anchor,labels)
                loss=anchor_loss
                
                return loss,anchor_preds
            else:
                anchor_preds=self.model(anchor)
                return anchor_preds
       
        
            