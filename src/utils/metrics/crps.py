import numpy as np 
import pandas as pd 
from scipy import special 
from numba import guvectorize
from typing import Union

# Reference list 
# https://pypi.org/project/properscoring/ 
# Evaluating Probabilistic Forecasts with scoringRules; Alexander Jordan, Fabian Krüger, Sebastian Lerch; Journal of Statistical Software (2019); 

def _standard_gaussian_dist_func(x): 
    """
    Distribution function of a univariate standard gaussian disribution.
    """
    return ((1.0) / np.sqrt(2.0 * np.pi)) * np.exp(-(x * x) / 2.0)


def _standard_gaussian_dens_func(x): 
    """ 
    Density function of a univaraite standard gaussian distribution. 
    """
    return special.ndtr(x) 


def crps_gaussian(mu, sigma, y):
    """
    CRPS for gaussian distribution with parameter (loc=mu, scale=sigma)

    Args:
        mu (_type_): loc parameter of gaussian distribution (forecast) 
        sigma (_type_): scale parameter of gaussian distribution (forecast)
        y (_type_): observed value 
    """
    stnd_ = (y - mu) / sigma
    pdf = _standard_gaussian_dist_func(stnd_) 
    cdf = _standard_gaussian_dens_func(stnd_) 
    pi_inv = 1.0 / np.sqrt(np.pi) 
    
    crps = sigma * (stnd_ * (2.0 * cdf - 1.0) + 2 * pdf - pi_inv) 
    dmu = 1 - 2 * cdf # gaussian의 경우 gradient를 미리 구해뒀으니 추후 사용 
    dsig = 2 * pdf - pi_inv 
    
    return crps 

def crps_laplace(mu, sigma, y):
    """
    CRPS for laplace distribution with parameter (loc=mu, scale=sigma)

    Args:
        mu (_type_): loc parameter of gaussian distribution (forecast) 
        sigma (_type_): scale parameter of gaussian distribution (forecast)
        y (_type_): observed value 
    """
    def _laplace(y):
        crps = np.abs(y) - np.exp(-np.abs(y)) - 3/4 
        return crps 

    stnd_ = (y - mu) / sigma 
    return sigma * _laplace(stnd_)  

def logistic_dens_func(x): 
    return 1.0 / (1+np.exp(-x))
        
def crps_logistic(mu, sigma, y): 
    """
    CRPS for logistic distribution with parameter (loc=mu, scale=sigma)

    Args:
        mu (_type_): loc parameter of gaussian distribution (forecast) 
        sigma (_type_): scale parameter of gaussian distribution (forecast)
        y (_type_): observed value 
    """
    def _logistic(y): 
        crps = y - 2 * np.log(logistic_dens_func(y)) - 1 
        return crps 
    
    stnd_ = (y - mu) / sigma 
    return sigma * _logistic(stnd_) 

    
@guvectorize(["void(float64[:], float64[:], float64[:])"], "(),(n)->()", nopython=True) # return void, input(float64 array x 3), 마지막은 result 
def _crps_numerical_method(observation, forecasts, result) -> Union[None, list]:
    obs = observation[0] # get observation 

    if np.isnan(obs):
        result[0] = np.nan # if obs == None return None 
        return

    # Count number of valid forecast 
    N = 0 
    for f in forecasts:
        if not np.isnan(f):
            N += 1
        else:
            break
    
    # if number of valid forecast == 0 return None 
    if N == 0:
        result[0] = np.nan
        return

    weight = 1.0 / N # Uniform weights -> 우리 문제에서는 보통 continuous value를 다루고 있으므로, cdf의 y값 분포는 uniform distribution이 됨. 
    #TODO: if discrete -> weight should vary

    obs_cdf = 0.0 # state of observed CDF 
    forecast_cdf = 0.0 # cummumlative sum of probability of forecast 
    prev_forecast = 0.0 # previous foercast 
    integral = 0.0 # crps value 

    for n in range(N): # for valid forecast 
        forecast = forecasts[n]

        if obs_cdf == 0.0 and obs < forecast: # The first iteration moment that forecast exceed observation. 
            integral += (obs - prev_forecast) * forecast_cdf ** 2 # left side of observation (observation > forecast)
            integral += (forecast - obs) * (forecast_cdf - 1) ** 2 # right side of observation (observaiotn < forecast) 
            obs_cdf = 1.0
        else:
            integral += (forecast - prev_forecast) * (forecast_cdf - obs_cdf) ** 2 # observation > forecast -> F(z) - 0; observation < forecast -> F(z) -1

        forecast_cdf += weight # cummulative summation of CDF 
        prev_forecast = forecast 

    if obs_cdf == 0.0: # obs < max(forecast) -> F(max(forecast)) = 1 therefore, \int^obs_max(forecast) (1-0)^2 d obs = x - max(forecast) 
        integral += obs - forecast # type: ignore

    result[0] = integral


def crps_numerical(y, forecast_samples, need_sorting=True): 
    """
    Numerical approximation of CRPS score 

    Args:
        y (_type_): _description_
        forecast_samples (_type_): _description_

    Returns:
        _type_: _description_
    """
    if need_sorting:
        sorted_forecasts = np.sort(forecast_samples, axis=-1) 
    else: 
        sorted_forecasts = forecast_samples
    result = np.empty(y.shape, dtype=np.float64)
    _crps_numerical_method(y, sorted_forecasts, result) 
    
    return np.mean(result) 