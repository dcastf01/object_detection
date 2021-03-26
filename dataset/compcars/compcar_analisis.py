import os
import sys
from typing import Union

import pandas as pd
from dataset.class_data_analisis import ClassDataAnalisis
from tqdm import tqdm
from dataset.compcars.enum_viepoints import ViewPointEnum

class CompcarAnalisis(ClassDataAnalisis):
    def __init__(self,
                 images_path:Union[str,None]=None,label_path:Union[str,None]=None,
                 path_csv:Union[str,None]=None) :         
        self.critical_variables=["viewpoint","viewpoint_name","make_id","model_id","released_year","model_id_and_released_year"]
        self.images_path=images_path
        self.label_path=label_path
        self.path_csv=path_csv
       
        df_temp=self.get_df_from_files()
        
            
        super().__init__(df_temp,name_dataset="compcars",critical_variables= self.critical_variables) 
    
    def get_df_from_files(self):

        def get_new_row_except_name_file(root):
            root_in_parts=root.split(os.sep)
            released_year=root_in_parts[-1]         
            model_id=root_in_parts[-2]
            make_id=root_in_parts[-3]
            new_row={ "released_year":released_year,
                    "model_id":model_id,
                    "make_id":make_id,      
                }
            return new_row

        def get_type_image(row):
            path_txt=os.path.join(self.label_path,row["make_id"],row["model_id"],row["released_year"],row["image_name"]+".txt")
            with open (path_txt,"r") as f:
                lines = f.readlines()
            
                #line 1 is the type of the image viewpoint annotation (-1 - uncertain, 1 - front, 2 - rear, 3 - side, 4 - front-side, 5 - rear-side)
                line1=lines[0].rstrip('\n')
                row["viewpoint"]=line1
                #line 2 is the number of bounding box
                line2=lines[1].rstrip('\n')
                row["nBoundingBox"]=line2
                #line 3 is is the coordinates of the bounding box in the format 'x1 y1 x2 y2' in pixels, where 1 <= x1 < x2 <= image_width, and 1 <= y1 < y2 <= image_height. 
                line3=lines[2].rstrip('\n')
                row["bbox"]=line3
                x1,y1,x2,y2=line3.split(" ")
                row["x1"]=x1
                row["y1"]=y1
                row["x2"]=x2
                row["y2"]=y2

            return row
        
        def create_dataframe_from_image_path(level_image=3):
            df=pd.DataFrame()


            for root, dirs, files in tqdm(os.walk(self.images_path),miniters=100):
                level = root.replace(self.images_path, '').count(os.sep)


                if level==level_image:
                    new_row=get_new_row_except_name_file(root)

                    for f in files:
                        filename=f.split(".")[0]
                        extension=f.split(".")[1]
                        new_row["image_name"]=filename
                        new_row["extension"]=extension
                        
                        df=df.append(new_row, ignore_index=True)
                    
            
            return df
        
        def set_type_data_compcars(df):
            return df.astype({'make_id': 'category',
                    "model_id":'category',
                    "released_year":"category",
                    "viewpoint":"category",
                    "nBoundingBox":"int32",
                    "x1":"int32",
                    "y1":"int32",
                    "x2":"int32",
                    "y2":"int32",
                    })
            
        tqdm.pandas(miniters=100)
        if self.path_csv is None:
            
            df=create_dataframe_from_image_path()
            df=df.progress_apply(lambda row: get_type_image(row),axis=1)
        else:
            
            df=pd.read_csv(self.path_csv,index_col =[0],dtype ='str')
        
        df=set_type_data_compcars(df)
        df["model_id_and_released_year"]=df["model_id"].astype(str)+df["released_year"].astype(str)
        df["viewpoint_name"] = df["viewpoint"].progress_apply(lambda x: ViewPointEnum(int(x)).name)
        return df
        
    
def test():
    path_csv=r"dataset\compcars\all_information_compcars.csv"
    compcar_analisis=CompcarAnalisis(path_csv=r"dataset\compcars\all_information_compcars.csv")
    # compcar_analisis=CompcarAnalisis(images_path=r"D:\programacion\Repositorios\object_detection_TFM\data\compcars\image",
    #                                  label_path=r"D:\programacion\Repositorios\object_detection_TFM\data\compcars\label")
    print("len dataset", len(compcar_analisis))
    compcar_analisis.filter_dataset('viewpoint=="4" or viewpoint=="1"')
    print("len dataset", len(compcar_analisis))
    
# test()






