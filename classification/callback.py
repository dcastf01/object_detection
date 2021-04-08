from pytorch_lightning.callbacks import Callback
import wandb

class ConfusionMatrix_Wandb(Callback):
    
    def __init__(self,class_names):
        super().__init__()
        self.class_names
    
    def on_validation_batch_end(self,trainer, pl_module,
                                outputs,
                                batch, batch_idx, 
                                dataloader_idx)->None
    
    
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=ground_truth, preds=predictions,
                            class_names=self.class_names
                    )})