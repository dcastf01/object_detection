import pandas as pd
import matplotlib.style as style
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p




import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def create_pareto_diagram( df,category):
  data=df[category].value_counts()
 
  df = pd.DataFrame({category: data})
  df = df.sort_values(by=[category],ascending=False)
  
  df["cumpercentage"] = df[category].cumsum()/df[category].sum()*100

  fig, ax = plt.subplots(figsize=(10,7))
  bar=df.plot(kind='bar',y=category,ax=ax,xlabel=category)
  
  ax2 = ax.twinx()
  line1=df.plot(kind='line',y="cumpercentage",ax=ax2,color="C1", marker="D", ms=7)
  ax2.yaxis.set_major_formatter(PercentFormatter())
  
  ax.tick_params(axis="y", colors="C0")
  ax2.tick_params(axis="y", colors="C1")
  ax2.set_ylim(ymin=0)
  ax.legend(loc="right")
  plt.xlabel(f"class {category}")
  plt.ylabel(f"unidades")

  plt.show()

def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
  
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histograma')
    ax1.set(xlabel=feature,ylabel=" cantidad de imágenes")

    ## plot the histogram. 
   # plt.hist(df.loc[:,feature], 20,)
    ax1.hist(df.loc[:,feature],30)
    
   # sns.distplot(df.loc[:,feature], norm_hist=False, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );