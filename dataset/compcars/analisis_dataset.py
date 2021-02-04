import os
import pandas as pd
from tqdm import tqdm

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
    

def get_df_from_files(images_path="/content/data/image",label_path="/content/data/label"):

    def get_new_row_except_name_file(images_path):
        root_in_parts=images_path.split("/")
        released_year=root_in_parts[-1]         
        model_id=root_in_parts[-2]
        make_id=root_in_parts[-3]
        new_row={ "released_year":released_year,
                "model_id":model_id,
                "make_id":make_id,      
            }
        return new_row

    def get_type_image(row):
        path_txt=os.path.join(label_path,row["make_id"],row["model_id"],row["released_year"],row["image_name"]+".txt")
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

    df=pd.DataFrame()


    for root, dirs, files in tqdm(os.walk(images_path)):
        level = root.replace(images_path, '').count(os.sep)


        if level==3:
            new_row=get_new_row_except_name_file(root)

            for f in files:
                filename=f.split(".")[0]
                extension=f.split(".")[1]
                new_row["image_name"]=filename
                new_row["extension"]=extension
                
                df=df.append(new_row, ignore_index=True)
            
            
    df=df.apply(lambda row: get_type_image(row),axis=1)
    
    df=set_type_data_compcars(df)
    
    return df





