

from dataset import config

from dataset.cars196.cars196_analisis import Cars196Analisis
from dataset.compcars.compcar_analisis import CompcarAnalisis


# cars196_analisis=Cars196Analisis(path_csv=config.PATH_CARS196_CSV)
# cars196_analisis.create_report_for_each_critical_variable()
compcar_analisis=CompcarAnalisis(path_csv=config.PATH_COMPCAR_CSV)

compcar_analisis.create_report_for_each_critical_variable()


