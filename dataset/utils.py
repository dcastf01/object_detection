from scipy.io import loadmat
from dataset import config

class CompcarsRealNames:
        
    def load_real_names_compcars(self):

        return loadmat(config.PATH_COMPCAR_MAKE_MODEL_NAME)

    def real_make_names_compcars(self): 
        return self.load_real_names_compcars()["make_names"]

    def get_real_make_name_with_index(self,index:int):
        return self.real_make_names_compcars()[index]
        
    def real_model_name_compcars(self):
        return self.load_real_names_compcars()["model_names"]
    
    def get_real_model_name_with_index(self,index:int):
        return self.real_model_name_compcars()[index]

def test():
    realnames=CompcarsRealNames()
    car_make_name=realnames.get_real_make_name_with_index(1)
    print("car_make_name",car_make_name)
    car_model_name=realnames.get_real_model_name_with_index(1)
    print("car_model_name",car_make_name)
test()