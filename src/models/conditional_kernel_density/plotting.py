import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np


def plot_power_pdf(density_estimates:pd.DataFrame, time:pd.Timestamp, observed_y:pd.Series):
    """ 
    Plot power density of a specific time. 
    
    Parameters 
    ---------- 
    density_estimates : pd.DataFrame
        The power density dataframe.
        index : pd.Timestamp 
    
    time : pd.Timestamp 
        The specific time to plot. 
    
    Returns
    -------
    None 
    """
    
    density_estimates_per_hour = density_estimates.loc[time] 
    plt.plot(np.linspace(0, 3600, 101), density_estimates_per_hour.values, label="Forecasted Wind Power Density") 
    plt.axvline(observed_y.loc[time], color="red", label="Observed Wind Power")
    plt.xlabel("Wind Power (KW)")
    plt.title(f"Estimated Wind Power PDF {time.strftime('%Y-%m-%d %H:%M:%S')}")
    plt.xlim(0, 3600)
    plt.xticks(np.linspace(0, 3600, 10))
    plt.show()



def plot_power_cdf(observed_y:pd.Series, density_estimates:pd.DataFrame, time:pd.Timestamp): 
    """ 
    Plot the CDF of the forecasted wind power with the observed wind power. 
    
    Parameters 
    ---------- 
    observed_y : pd.Series 
        The observed wind power values. 
    density_estimates : pd.DataFrame 
        The density estimates for each hour. 
    time : pd.Timestamp 
        The time to plot the CDF for. 
        
    Returns 
    ------- 
    None 
    """
    density_estimates_per_hour = density_estimates.loc[time] 
    density_estimates_per_hour = density_estimates_per_hour / np.sum(density_estimates_per_hour)
    cdf = np.cumsum(density_estimates_per_hour) 
    
    plt.plot(np.linspace(0, 3600, 101), cdf, label="Forecasted CDF")
    plt.axvline(observed_y.loc[time], color="red", label="Observed Wind Power")
    plt.xlabel("Wind Power (KW)")
    plt.title(f"Estimated Wind Power CDF {time.strftime('%Y-%m-%d %H:%M:%S')}")
    plt.xlim(0, 3600)
    plt.ylim(0)
    plt.xticks(np.linspace(0, 3600, 10))
    plt.legend()
    plt.show()
