from dataset import config

from dataset.cars196.cars196_analisis import Cars196Analisis
from dataset.compcars.compcar_analisis import CompcarAnalisis


def choice_dataset(name_dataset:str):
    
    if name_dataset.lower()=="cars196":
        return Cars196Analisis(path_csv=config.PATH_CARS196_CSV)
    elif name_dataset.lower() =="compcars":
        return CompcarAnalisis(path_csv=config.PATH_COMPCAR_CSV)


cars196_analisis=choice_dataset("cars196")
cars196_analisis.create_report_for_each_critical_variable()

compcar_analisis=choice_dataset("compcars")
compcar_analisis.create_report_for_each_critical_variable()
compcar_analisis.filter_dataset('viewpoint=="4" or viewpoint=="1"')
compcar_analisis.create_report_for_each_critical_variable(add_text_on_title= "subset front and front-side")


