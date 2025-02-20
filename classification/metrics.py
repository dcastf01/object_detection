#https://torchmetrics.readthedocs.io/en/stable/pages/overview.html

#echar un vistazo https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226
#https://pytorch-lightning.medium.com/torchmetrics-pytorch-metrics-built-to-scale-7091b1bec919
import torch
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall,StatScores

def get_metrics_collections_base(NUM_CLASS,
                            # device="cuda" if torch.cuda.is_available() else "cpu",
                            
                            ):
    
    metrics = MetricCollection(
            {
                "Accuracy":Accuracy(),
                "Top_3":Accuracy(top_k=3),
                "Top_5" :Accuracy(top_k=5),
                "Precision_micro":Precision(num_classes=NUM_CLASS,average="micro"),
                "Precision_macro":Precision(num_classes=NUM_CLASS,average="macro"),
                "Recall_micro":Recall(num_classes=NUM_CLASS,average="micro"),
                "Recall_macro":Recall(num_classes=NUM_CLASS,average="macro"),
                "F1_micro":torchmetrics.F1(NUM_CLASS,average="micro"),
                "F1_macro":torchmetrics.F1(NUM_CLASS,average="micro"),
            }
            )
    
    
    return metrics

def get_metric_AUROC(NUM_CLASS):
    
    metrics=MetricCollection(
        {
                "AUROC_macro":torchmetrics.AUROC(num_classes=NUM_CLASS)
                        
        }
    )
    return metrics