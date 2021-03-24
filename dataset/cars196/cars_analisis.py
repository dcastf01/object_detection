
from typing import Union

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from dataset.cars196 import cars196
from dataset.class_data_analisis import ClassDataAnalisis
from tqdm import tqdm


class Cars196Analisis(ClassDataAnalisis):
    def __init__(self,
                 images_path:Union[str,None]=None,#r"data\cars196" ,
                 data_annotations_path_train:Union[str,None]=r"data\cars196\extracted\TAR_GZ.ai.stanford.edu_jkrause_cars_car_devkituX3rRjr31Ytr-qGLKk3pgp8PeejOZj36kmG_eBDprM0.tgz\devkit\cars_train_annos.mat",
                 data_annotations_path_test:Union[str,None]=r"data\cars196\extracted\TAR_GZ.ai.stanford.edu_jkrause_cars_car_devkituX3rRjr31Ytr-qGLKk3pgp8PeejOZj36kmG_eBDprM0.tgz\devkit\cars_test_annos.mat",
                 path_csv:Union[str,None]=None) :     
        
            
        
        self.data_annotations_path_train=data_annotations_path_train
        self.data_annotations_path_test=data_annotations_path_test
        self.critical_variables=["make_id","model_id","released_year"]
        self.images_path=images_path
        # self.label_path=label_path
        self.path_csv=path_csv
       
        df_temp=self.get_df_total_train_and_test()
        
            
        super().__init__(df_temp, self.critical_variables) 
    def __len__(self):
        return self.data.shape[0]
    def create_dataframe_cars196_by_annotations(self,data_annotations):
        df=pd.DataFrame()


        with tf.io.gfile.GFile(data_annotations, 'rb') as f:
            mat = tfds.core.lazy_imports.scipy.io.loadmat(f)
            for example in tqdm(mat['annotations'][0],miniters=100):
                image_name = example[-1].item().split('.')[0]
                label = cars196._NAMES[example[4].item() - 1]

                features = {
                        'label': label,
                        'filename':str(image_name),
                    }
                df=df.append(features,ignore_index=True)  
        return df

    def get_df_total_train_and_test(self):
        if self.path_csv is not None:
            df=pd.read_csv( self.path_csv,index_col =[0],dtype ='str')
        else:
            
            if self.images_path is not  None:
                cars196.Cars196v2()
                cars_builder=tfds.builder("cars196v2")
                cars_builder.download_and_prepare(download_dir=self.images_path)
            df=self.create_dataframe_cars196_by_annotations(self.data_annotations_path_train)
            # df.filename.astype(str)
            df.to_csv(r"dataset\cars196\all_information_cars196.csv")
            # df_test=self.create_dataframe_cars196_by_annotations(self.data_annotations_path_test)
            
            # df=pd.concat([df_train,df_test],)
            
        df["make_id"]=df.label.str.split(" ").str[0]
        df["model_id"]=df.label.str.split(" ").str[1:-1].str.join(" ")
        df["released_year"]=df.label.str.split(" ").str[-1]
        
        print(df.head())
        return df
        
def test():
    # path_csv=r"dataset\compcars\all_information_compcars.csv"
    cars196_analisis=Cars196Analisis(path_csv= r"dataset\cars196\all_information_cars196.csv")
    # cars196_analisis=Cars196Analisis()
    print("len dataset", len(cars196_analisis))
    
test()