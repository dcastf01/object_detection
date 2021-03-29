from scipy.io import loadmat
import config

class CompcarsRealNames:
    def __init__(self):
        self.real_names_makes=self.load_real_names_compcars()["make_names"]
        self.real_names_models=self.load_real_names_compcars()["model_names"]
        
    def load_real_names_compcars(self):

        return loadmat(config.PATH_COMPCAR_MAKE_MODEL_NAME)

    def real_make_names_compcars(self): 
        return self.load_real_names_compcars()["make_names"]

    def get_real_make_name_with_index(self,index:int):
        return self.real_names_makes[index]
        
    def real_model_name_compcars(self):
        return self.load_real_names_compcars()["model_names"]
    
    def get_real_model_name_with_index(self,index:int):
        return self.real_names_models[index]

def test_():
    realnames=CompcarsRealNames()
    car_make_name=realnames.get_real_make_name_with_index(1)
    
    print("car_make_name",car_make_name)
    car_model_name=realnames.get_real_model_name_with_index(1)
    print("car_model_name",car_model_name)
# test()