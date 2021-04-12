
import pandas as pd

from torch.utils.data import Dataset
class Loader (Dataset):
    
    def __init__ (self, df:pd.DataFrame,root_dir_images:str,
                  transform=None,condition_filter:str=None,
                  ):
        self.data=df
        self.root_dir_images=root_dir_images
        
        self.transform=transform

        if condition_filter is not None:
            self.condition_filter=condition_filter
            self.data=self.data.query(condition_filter)
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,index):
        
        raise NotImplementedError