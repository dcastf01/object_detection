import torch

#dataset compcar
PATH_COMPCAR_CSV=r"dataset\compcars\all_information_compcars.csv"
PATH_COMPCAR_IMAGES=r"data\compcars\image"
PATH_COMPCAR_LABELS=r"data\compcars\label"
PATH_COMPCAR_MAKE_MODEL_NAME=r"data\compcars\misc\make_model_name.mat"
COMPCAR_CONDITION_FILTER='viewpoint=="4" or viewpoint=="1"'

#dataset cars196
PATH_CARS196_CSV=r"dataset\cars196\all_information_cars196.csv"
# PATH_CARS196_IMAGES=r"dataset\compcars\all_information_compcars.csv"
# PATH_CARS196_LABELS=r"dataset\compcars\all_information_compcars.csv"

#output plots
PATH_OUTPUT_PLOTS=r"dataset\results"


#torch config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TRAIN_DIR = "data/train"
# VAL_DIR = "data/val"
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
# LAMBDA_IDENTITY = 0.0
# NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"