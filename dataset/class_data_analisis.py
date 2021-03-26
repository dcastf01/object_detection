import pandas as pd
from dataset import config
from dataset.charts import create_pareto_diagram,plotting_3_chart
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
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
        self.filter=None
        
    def __len__(self):
        return self.data.shape[0]
        
    def filter_dataset(self,condition):
        self.filter=condition
        self.data=self.data.query(condition)
        
    def create_report(self,variable)->plt.figure:
        return create_pareto_diagram(self.data,variable)
        
    def create_report_for_each_critical_variable(self,add_text_on_title:str=""):
        logging.info(f"do reports from {self.name_dataset}")
        os.makedirs(self.output_result_plots, exist_ok=True)
        for critical_variable in tqdm(self.critical_variables):
            fig=self.create_report(critical_variable)
            
            fig.suptitle(f"dataset: {self.name_dataset} {add_text_on_title}", fontsize=16)
            plt.title(f"variable: {critical_variable}")
            # fig.set_subtitle(f"variable: {critical_variable}")
            plt.tight_layout()
            try:
                fig.savefig(os.path.join(self.output_result_plots,critical_variable+"_"+self.name_dataset+"_"+add_text_on_title))
            except Exception as e:
                print (e)
            
            
            
            

