from scipy.io import loadmat

path_data="/home/dcast/object_detection_TFM/data/compcars/CompCars_revisited_v1.0/results.mat"
data=loadmat(path_data)["None"][0]
a=data.res_hml
for _ in data:
    print( _)