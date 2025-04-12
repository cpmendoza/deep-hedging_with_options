"""
Usage:
    1. cd src
    2. python3 features/nig_simulation.py 
"""

import os, sys
from pathlib import Path
import warnings

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import norminvgauss
from src.utils import *

def nig_simulation(config_file, root_directory = main_folder):

    """
    Inputs:
    config_file     : Simulation settings 
    Outputs:
    NIG_simulation  : NIG random values for each for the JIVR model from François et al. (2023).
                      This numpy array is stored in ../data/interim
    """

    number_simulations = config_file['simulation']['number_simulations']*2 #Multiply by two just to have a lager sample size
    time_steps = config_file['simulation']['number_days']
    seed = config_file['simulation']['seed']

    # Set the seed to a specific value (e.g., 42)
    np.random.seed(seed)

    #Generation of NIG values with Gaussian copula, reference: François et al. (2023).
    directory = os.path.join(root_directory,'data','raw','Parameters_2.csv')
    parameters = pd.read_csv(directory, index_col=0)

    #Normal random values con copula parameters
    cov_mat = np.array([1.00, -0.55, -0.69, 0.03, -0.22, -0.34, -0.55, 1.00, 0.14, -0.03, 0.25, 0.17, -0.69, 0.14, 1.00, -0.01, 0.12, 0.37, 0.03, -0.03, -0.01, 1.00, 0.28, 0.13, -0.22, 0.25, 0.12, 0.28, 1.00, -0.05, -0.34, 0.17, 0.37, 0.13, -0.05, 1.00]).reshape([6,6])
    mean = [0,0,0,0,0,0]
    normal_estandar = np.random.multivariate_normal(mean, cov_mat, time_steps*number_simulations)
    normal_estandar = normal_estandar[:,[1,2,3,4,5,0]]

    #Generation of uniform random values from Gaussian copula
    uniform = np.maximum(np.minimum(norm.cdf(normal_estandar),0.9995),0.0001)

    #Modify parameters from the standard NIG used in François et al. (2023) to obtain the standard NIG parametrization
    a_hat = 3*parameters.iloc[12,:]/(parameters.iloc[13,:]**2)
    b_hat = 3*((parameters.iloc[13,:]**2+5*(parameters.iloc[12,:]**2)))/(parameters.iloc[13,:]**4)

    beta_1 = a_hat/(b_hat-(5/3)*a_hat**2)
    alpha_1 = np.sqrt((3/a_hat)*beta_1+beta_1**2)
    delta_1 = alpha_1*(np.sqrt(1-(beta_1/alpha_1)**2)**3)
    mu_1 = -1*(delta_1*beta_1)/(alpha_1*np.sqrt(1-(beta_1/alpha_1)**2))
    #Standard parametrization
    a = alpha_1*delta_1
    b = beta_1*delta_1
    c = mu_1
    d = delta_1

    #Generate NIG grid for interpolation
    print("--Generating grid for NIG interpolation--")
    grid_division = 1000
    uniform_grid = np.linspace(0.0001, 0.9995, num=grid_division)
    NIG_grid = {}
    for j in range(6):
        NIG_grid[j]=norminvgauss.ppf(uniform_grid, a[j], b[j], loc=c[j], scale=d[j])

    print("--NIG simulation starts--")
    print("--Progress of NIG simulation: ", end='', flush=True)
    NIG = np.zeros(uniform.shape)
    for i in range(time_steps):
        subset = range(i*number_simulations,(i+1)*number_simulations)
        for j in range(6):
            auxiliary_array = np.repeat(uniform[subset,j].reshape(-1,1), grid_division, axis=1)
            index_upper = np.argmax((auxiliary_array < uniform_grid)*1,axis=1)
            upper_limit_prob = uniform_grid[index_upper]
            lower_limit_prob = uniform_grid[index_upper-1]
            upper_limit_nig = NIG_grid[j][index_upper]
            lower_limit_nig = NIG_grid[j][index_upper-1]
            #Linear interpolation
            simulation = ((upper_limit_nig - lower_limit_nig)/(upper_limit_prob - lower_limit_prob))*(uniform[subset,j]-lower_limit_prob)+lower_limit_nig
            NIG[subset,j] = simulation
        print("\r--Progress of NIG simulation: {:.2%}".format((i+1)/63), end='', flush=True)

    np.save(os.path.join(root_directory,'data','interim','NIG_simulation.npy'),NIG)
    print("\n--Simulation completed - NIG values stored in ../data/interim/--")

    return 

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file = load_config(os.path.join(main_folder,'cfgs','config_simulation.yml'))
    results = nig_simulation(config_file)
    
