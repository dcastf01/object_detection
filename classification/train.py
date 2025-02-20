#check this https://stackoverflow.com/questions/64607182/vs-code-remote-ssh-how-to-allow-processes-to-keep-running-to-completion-after-d
import datetime
import logging
# os.environ["PYTHONPATH"] ='/home/dcast/object_detection_TFM'
import sys
sys.path.append("/home/dcast/object_detection_TFM")
import pytorch_lightning as pl
import torch
import wandb
from config import CONFIG,create_config_dict
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from classification.callback import ConfusionMatrix_Wandb
from classification.choice_loader import choice_loader_and_splits_dataset,Dataset
from classification.lit_classifier import LitClassifier
from classification.model.build_model import build_model
from classification.autotune import autotune_lr

    
def main():
    print("empezando setup del experimento")
    torch.backends.cudnn.benchmark = True
    config=CONFIG()
    config_dict=create_config_dict(config)
    wandb.init(
        project='TFM-classification',
                entity='dcastf01',
                name=config.experiment_name+" "+
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %X"),
                    
                config=config_dict)
    
    wandb_logger = WandbLogger(
                    # offline=True,
                    )

    config =wandb.config
    dataloaders,NUM_CLASSES=choice_loader_and_splits_dataset(
                                                config.dataset_name,
                                                batch_size=config.batch_size,
                                                NUM_WORKERS=config.NUM_WORKERS,
                                                use_tripletLoss=config.USE_TRIPLETLOSS,#config_experiment.use_tripletLoss
                                                )
    
    logging.info("DEVICE",config.DEVICE)
    train_loader=dataloaders["train"]
    test_loader=dataloaders["test"]
    

    ##callbacks
    early_stopping=EarlyStopping(monitor='_val_loss',verbose=True)
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='_val_loss',
    #     dirpath=config.PATH_CHECKPOINT,
    #     filename= '-{epoch:02d}-{val_loss:.6f}',
    #     mode="min",
    #     save_last=True,
    #     save_top_k=3,
    #                     )
    learning_rate_monitor=LearningRateMonitor(logging_interval="epoch")
    
        
    backbone=build_model(   config.experiment_name,
                        NUM_CLASSES=NUM_CLASSES,
                        pretrained=config.PRETRAINED_MODEL,
                        transfer_learning=config.transfer_learning,

                            
                        )
    model=LitClassifier(backbone,
                    # loss_fn=loss_fn,
                    lr=config.lr,
                    NUM_CLASSES=NUM_CLASSES,
                    optim=config.optim_name
                    )
    # model=model.model.load_from_checkpoint("/home/dcast/object_detection_TFM/classification/model/checkpoint/last.ckpt")
    wandb_logger.watch(model.model)
    trainer=pl.Trainer(
                        logger=wandb_logger,
                       gpus=-1,
                       max_epochs=config.NUM_EPOCHS,
                       precision=config.precision_compute,
                    #    limit_train_batches=0.1, #only to debug
                    #    limit_val_batches=0.05, #only to debug
                    #    val_check_interval=1,
                        auto_lr_find=config.AUTO_LR,

                       log_gpu_memory=True,
                    #    distributed_backend='ddp',
                    #    accelerator="dpp",
                    #    plugins=DDPPlugin(find_unused_parameters=False),
                       callbacks=[
                            # early_stopping ,
                            # checkpoint_callback,
                            # confusion_matrix_wandb,
                            learning_rate_monitor 
                                  ],
                       progress_bar_refresh_rate=5,
                       )
    
    model=autotune_lr(trainer,model,test_loader,get_auto_lr=config.AUTO_LR)
    logging.info("empezando el entrenamiento")
    trainer.fit(model,train_loader,test_loader)
         

if __name__ == "__main__":
    
    main()
