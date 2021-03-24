import pandas as pd
from dataset import config
from dataset.charts import create_pareto_diagram,plotting_3_chart
import os
import matplotlib.pyplot as plt

class ClassDataAnalisis:
    def __init__(self,df:pd.DataFrame,
                 name_dataset:str,
                 critical_variables:list,
                 output_result_plots:str=config.PATH_OUTPUT_PLOTS):
        self.data=df
        self.name_dataset=name_dataset
        self.critical_variables=critical_variables
        
        self.root_path=os.getcwd()
        self.output_result_plots=os.path.join(self.root_path,output_result_plots)
        
        
        
        
    def create_report(self,variable)->plt.figure:
        return create_pareto_diagram(self.data,variable)
        
    def create_report_for_each_critical_variable(self):
        os.makedirs(self.output_result_plots, exist_ok=True)
        for critical_variable in self.critical_variables:
            fig=self.create_report(critical_variable)
            
            fig.suptitle(f"dataset: {self.name_dataset}", fontsize=16)
            plt.title(f"variable: {critical_variable}")
            # fig.set_subtitle(f"variable: {critical_variable}")
            plt.tight_layout()
            try:
                fig.savefig(os.path.join(self.output_result_plots,critical_variable+"_"+self.name_dataset))
            except Exception as e:
                print (e)
            
            
            
            

