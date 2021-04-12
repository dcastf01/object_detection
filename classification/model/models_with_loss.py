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
            
            return {"loss":loss,
                    "preds":preds}
        else:
 
            return {"preds":anchor_preds}
        
        
class ModelWithTripletLoss(nn.Module):
    
    def __init__(self,model):
        super(ModelWithTripletLoss,self).__init__()
        
        self.model=model
        self.triplet_loss=nn.TripletMarginLoss()
    
    def get_loss_and_preds(self,imgs,labels):
        
        aux_dict=self.model(imgs,labels)
        return aux_dict["loss"],aux_dict["preds"]
        
    def forward(self,x,labels=None):
        if len(x)==3:
            anchor_imgs,positive_imgs,negative_imgs=x
            anchor_labels,positive_labels,negative_labels=labels
            
            if labels is not None:
                
                anchor_loss,anchor_preds=self.get_loss_and_preds(anchor_imgs,anchor_labels)
                
                positive_loss,positive_preds=self.get_loss_and_preds(positive_imgs,positive_labels)
        
                negative_loss,negative_preds=self.get_loss_and_preds(negative_imgs,negative_labels)
              
                
                #me he sacado de la manga que se sume todas las funciones de perdida 
                #pero a lo mejor no tiene sentido
                loss_original_from_models=torch.sum(torch.stack([anchor_loss, positive_loss, negative_loss]), dim=0)
                loss_triplet=self.triplet_loss(anchor_preds,positive_preds,negative_preds)
                
                loss=loss_triplet+loss_original_from_models/3
                return {"loss":loss,
                        "preds":anchor_preds,
                        "loss_triplet":loss_triplet,
                        "loss_original_from_models":loss_original_from_models}
            else:
                anchor_preds=self.model(anchor)
                return {"preds":anchor_preds}
        else:
            anchor=x
            if labels is not None:
                data_dict=self.model(anchor,labels)
                
                return data_dict
            else:
                anchor_preds=self.model(anchor)
                return {"preds":anchor_preds}
       
        
            