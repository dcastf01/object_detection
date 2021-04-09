import torch
import os
class CONFIG:
    ROOT_WORKSPACE=r"D:\programacion\Repositorios\object_detection_TFM"
    #dataset compcar
    PATH_ROOT_COMPCAR=os.path.join(ROOT_WORKSPACE,"data","compcars")
    PATH_COMPCAR_CSV=os.path.join(PATH_ROOT_COMPCAR,"all_information_compcars.csv")
    PATH_COMPCAR_IMAGES=os.path.join(PATH_ROOT_COMPCAR,"image")
    PATH_COMPCAR_LABELS=os.path.join(PATH_ROOT_COMPCAR,"label")
    PATH_COMPCAR_TRAIN_REVISITED=os.path.join(PATH_ROOT_COMPCAR,"CompCars_revisited_v1.0","bbs_train.txt")
    PATH_COMPCAR_TEST_REVISITED=os.path.join(PATH_ROOT_COMPCAR,"CompCars_revisited_v1.0","bbs_test.txt")
    PATH_COMPCAR_MAKE_MODEL_NAME=os.path.join(PATH_ROOT_COMPCAR,"misc","make_model_name.mat")
    PATH_COMPCAR_MAKE_MODEL_NAME_CLS=os.path.join(PATH_ROOT_COMPCAR,"misc","make_model_names_cls.mat")
    COMPCAR_CONDITION_FILTER='viewpoint=="4" or viewpoint=="1"'

    #dataset cars196
    PATH_CARS196_CSV=os.path.join(ROOT_WORKSPACE,r"dataset\cars196\all_information_cars196.csv")
    # PATH_CARS196_IMAGES=r"dataset\compcars\all_information_compcars.csv"
    # PATH_CARS196_LABELS=r"dataset\compcars\all_information_compcars.csv"

    #output plots
    PATH_OUTPUT_PLOTS=os.path.join(ROOT_WORKSPACE,r"dataset\results")


    #torch config
    
        

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # TRAIN_DIR = "data/train"
    # VAL_DIR = "data/val"
    BATCH_SIZE = 16
    NUM_CLASSES=4455
    LEARNING_RATE = 1e-5
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS = 0
    SEED=1

    NUM_EPOCHS = 10
    LOAD_MODEL = True
    SAVE_MODEL = True

    PATH_CHECKPOINT= os.path.join(ROOT_WORKSPACE,"classification/model/checkpoint")

    class ModelName:
        torch_squeezenet="torch_squeezeNet"
        torch_transFG="torch_transFG"
