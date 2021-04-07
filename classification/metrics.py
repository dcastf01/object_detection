#https://torchmetrics.readthedocs.io/en/stable/pages/overview.html

#echar un vistazo https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226
import torch
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall,StatScores

def get_metrics_collections(NUM_CLASS,device="cuda" if torch.cuda.is_available() else "cpu"):
    
    metrics = MetricCollection(
            [
                Accuracy(),
                Precision(),
                Recall(),
                StatScores(),
                torchmetrics.F1(NUM_CLASS),
                # torchmetrics.ConfusionMatrix(NUM_CLASS),
                
            ]
            ).to(device)
    
    
    return metrics