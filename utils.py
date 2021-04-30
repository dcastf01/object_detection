import cv2
import pandas as pd
import os
from config import CONFIG
from tqdm import tqdm

def crop_and_save_images(DATA_ROOT,img_path,x1,x2,y1,y2):
    def crop_image(img,x1,x2,y1,y2):
        cropped_im=img[y1:y2, x1:x2, :]
        return cropped_im
    
    img=cv2.imread(os.path.join(DATA_ROOT, 'image', img_path))
    cropped_img=crop_image(img,x1,x2,y1,y2)
    
    make, model, year, filename = img_path.replace('jpg', 'txt').split('/')
    if not os.path.exists(os.path.join(DATA_ROOT, 'cropped_image', make, model, year)):
        try:
            os.makedirs(os.path.join(DATA_ROOT, 'cropped_image', make, model, year))
        except:
            pass
    
    cv2.imwrite(os.path.join(os.path.join(DATA_ROOT, 'cropped_image', img_path)), cropped_img)
    
def crop_and_save_compcarimages():

    def expand_information_useful_on_txt(df:pd.DataFrame)->pd.DataFrame:
                df[["make_id","model_id","released_year","filename"]]=df["Filepath"].str.split("/",expand=True)
                return df  
    
    def get_img_path_crop_and_save(df:pd.DataFrame,DATA_ROOT):
        
        for i,row in tqdm(df.iterrows()):
            
            img_path=row["Filepath"]
            x1=row["X"]
            y1=row["Y"]
            
            x2=row["X"]+row["Width"]
            y2=row["Y"]+row["Height"]
            crop_and_save_images(DATA_ROOT,img_path,x1,x2,y1,y2)
            
    
    train_ds = pd.read_csv(CONFIG.DATASET.COMPCAR.PATH_TRAIN_REVISITED)
    train_ds=expand_information_useful_on_txt(train_ds)

    test_ds=pd.read_csv(CONFIG.DATASET.COMPCAR.PATH_TEST_REVISITED,)
    test_ds=expand_information_useful_on_txt(test_ds)
    # get_img_path_crop_and_save(train_ds,CONFIG.DATASET.COMPCAR.PATH_ROOT)
    get_img_path_crop_and_save(test_ds,CONFIG.DATASET.COMPCAR.PATH_ROOT)
    
    

crop_and_save_compcarimages()