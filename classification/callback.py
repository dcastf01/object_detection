from pytorch_lightning.callbacks import Callback
import wandb
from config import CONFIG
import torch
class ConfusionMatrix_Wandb(Callback):
    
    def __init__(self,class_names):
        super().__init__()
        self.class_names=class_names
    
    def on_validation_batch_end(self,trainer, pl_module,
                                outputs,
                                batch, batch_idx, 
                                dataloader_idx)->None:
    
        x, targets = batch
        preds = pl_module.model(x.to(CONFIG.DEVICE)).softmax(dim=-1)
        preds=torch.argmax(preds,dim=1)
        
        
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=targets, preds=preds,
                            class_names=self.class_names
                    )})