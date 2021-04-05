from scipy.io import loadmat
from config import CONFIG
import numpy as np
import os 
import sys


class CompcarsRealNames:
    def __init__(self):
        self.real_names_makes=self.load_real_names_compcars()["make_names"]
        self.real_names_models=self.load_real_names_compcars()["model_names"]
    
    def load_make_models_CLS_names(self):
        return loadmat(config.PATH_COMPCAR_MAKE_MODEL_NAME_CLS,)
    
    def load_real_names_compcars(self):

        return loadmat(config.PATH_COMPCAR_MAKE_MODEL_NAME,squeeze_me=True)
        

    def real_make_names_compcars(self): 
        return self.load_real_names_compcars()["make_names"]

    def get_real_make_name_with_index(self,index:int):
        return self.real_names_makes[index]
        
    def real_model_name_compcars(self):
        return self.load_real_names_compcars()["model_names"]
    
    def get_real_model_name_with_index(self,index:int):
        return self.real_names_models[index]

def test():
    realnames=CompcarsRealNames()
    car_make_name=realnames.get_real_make_name_with_index(1)
    
    print("car_make_name",car_make_name)
    car_model_name=realnames.get_real_model_name_with_index(1)
    print("car_model_name",car_model_name)
    all_car_model_name=realnames.real_model_name_compcars()
    all_car_model_name=np.array(all_car_model_name)
    print(all_car_model_name.shape)
    print(all_car_model_name)
    print(len(all_car_model_name))
test()