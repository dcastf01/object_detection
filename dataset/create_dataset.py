import pandas as pd
def create_dataframe_from_image_path(images_path,level_image=3):
    df=pd.DataFrame()


    for root, dirs, files in tqdm(os.walk(images_path)):
        level = root.replace(images_path, '').count(os.sep)


        if level==level_image:
            new_row=get_new_row_except_name_file(root)

            for f in files:
                filename=f.split(".")[0]
                extension=f.split(".")[1]
                new_row["image_name"]=filename
                new_row["extension"]=extension
                
                df=df.append(new_row, ignore_index=True)
            
    
    return df