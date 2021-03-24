import pandas as pd

from dataset.analisis_dataset import create_pareto_diagram,plotting_3_chart

class ClassDataAnalisis:
    def __init__(self,df:pd.DataFrame,critical_variables:list):
        self.data=df
        self.critical_variables=critical_variables
        
    def create_report(self,variable):
        create_pareto_diagram(self.data,variable)
        
    def create_report_for_each_critical_variable(self):
        for critical_variable in self.critical_variables:
            self.create_report(critical_variable)
            
            

