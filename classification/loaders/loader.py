
import pandas as pd

from torch.utils.data import Dataset
import os
class Loader (Dataset):
    
    def __init__ (self, df:pd.DataFrame,root_dir_images:str,
                  transform=None,condition_filter:str=None,
                  ):
        self.data=df
        root_dir=root_dir_images.split(os.path.sep)[:-1]
        root_dir_crooped_image=os.path.join(*root_dir,"cropped_image")
        if os.path.exists(root_dir_crooped_image):
            self.root_dir_images=root_dir_crooped_image
        else:
            self.root_dir_images=root_dir_images
        
        self.transform=transform

        if condition_filter is not None:
            self.condition_filter=condition_filter
            self.data=self.data.query(condition_filter)
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,index):
        
        raise NotImplementedError