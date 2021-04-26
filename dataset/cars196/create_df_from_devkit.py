import pandas as pd
from scipy.io import loadmat
def create_df_from_devkit(path):
  info=loadmat(path)
  image_names=[]
  extensions=[]
  labels=[]
  make_ids=[]
  model_ids=[]
  released_years=[]
  ymins=[]
  xmins=[]
  ymaxs=[]
  xmaxs=[]
  viewpoints=[]
  nBoundingBoxs=[]
  bboxs=[]

  for example in info['annotations'][0]:
    # print(example)
    
    image_name = example[-1].item().split('.')[0]
    extension=example[-1].item().split('.')[-1]
    label = _NAMES[example[4].item() - 1]
    make_id=label.split(" ")[0]
    model_id=" ".join(label.split(" ")[1:-1])#.str.join(" "))
    released_year=label.split(" ")[-1]
    ymin = float(example[1].item())
    xmin = float(example[0].item())
    ymax = float(example[3].item())
    xmax = float(example[2].item())
    viewpoint="nan"
    nBoundingBox=1
    bbox=" ".join([str(int(xmin)),str(int(ymin)),str(int(xmax)),str(int(ymax))])
    
    image_names.append(image_name)
    extensions.append(extension)
    labels.append(label)
    make_ids.append(make_id)
    model_ids.append(model_id)
    released_years.append(released_year)
    ymins.append(ymin)
    xmins.append(xmin)
    ymaxs.append(ymax)
    xmaxs.append(xmax)
    viewpoints.append(viewpoint)
    nBoundingBoxs.append(nBoundingBox)
    bboxs.append(bbox)

  data=pd.DataFrame(list(zip(image_names,
                            extensions,
                            labels,
                            make_ids,
                            model_ids,
                            released_years,
                            ymins,
                            xmins,
                            ymaxs,
                            xmaxs,
                            viewpoints,
                            nBoundingBoxs,
                            bboxs
                            )
                      ),
                    columns=[
                            "image_name",
                            "extension",
                            "labels",
                            "make_ids",
                            "model_ids",
                            "released_years",
                            "ymins",
                            "xmins",
                            "ymaxs",
                            "xmaxs",
                            "viewpoints",
                            "nBoundingBoxs",
                            "bboxs"

                            
                        ]
                      )
  return data