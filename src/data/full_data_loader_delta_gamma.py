"""
Usage:
    1. cd src
    2. python data/data_loader.py
"""

import os, sys
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os, sys
import numpy as np
import math

from pathlib import Path
from scipy.stats import norm
from src.features.market_simulator import *
from src.visualization.strategy_evaluation import delta_gamma


def data_sets_preparation(temporality, S, B, H, hedged_options_maturities, hedged_option_types, hedged_moneyness, hedged_positions, hedging_intruments_maturity, hedging_option_types, hedging_option_moneyness, dynamics, lower_bound, r, q, delta, tc, issmile, trainphase):

    """Function that loads the sets to create the training set

        Parameters
        ----------
        temporality                  : Temporality of the simulation
        r                            : Risk-free rate                  
        q                            : Dividend yield rate
        delta                        : Step-size (daily rebalancing)
        tc                           : Transaction cost level
        trainphase                   : Simulation based on real or simulated data
        lower_bound                  : Lower bound to clip IV surface values
        issmile                      : To determine the type of greeks 
        hedged_options_maturities    : Option maturities for instruments to hedge
        hedged_option_types          : Option types {Call: False, Put: True}
        hedged_moneyness             : Option moneyness for instruments to hedge {"ATM","OTM","ITM"}
        hedged_positions             : Number of shares of each option to hedge
        hedging_intruments_maturity  : Option maturities for hedging instruments
        hedging_option_types         : Option types {Call: False, Put: True}
        hedging_option_moneyness     : Option moneyness for hedging instruments {"ATM","OTM","ITM"}
        dynamics                     : Simulation type {Single intrument: 'static', New instrument every day: 'dynamic'}
        
        Returns
        -------
        id                   : Acronym for the hedging problem
        train_input          : Training set (normalized stock price and features)
        test_input           : Test set (normalized stock price and features)
        HO_train             : Prices of hedging instruments in the training set
        HO_test              : Prices of hedging instruments in the validation set
        cash_flows_train     : Cash flows of hedged portfolio for training set
        cash_flows_test      : Cash flows of hedged portfolio for validation set
        risk_free_factor     : Risk-free rate update factor exp(h*r)
        dividendyield_factor : Dividend yield update factor exp(h*d)
        underlying_train     : Underlying asset prices for training set
        underlying_test      : Underlying asset prices for validation set

      """

    #Identify number of simulations
    number_simulations = S.shape[0]
    n_timesteps        = temporality+1
    h                  = delta

    #ID hedged instruments
    id = "HO_" #Hedged options
    for i in range(len(hedged_options_maturities)):
        if hedged_option_types[i] == False:
            id += 'C' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 
        else:
            id += 'P' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 

    if len(hedging_intruments_maturity)!=0:
    #ID hedigng instruments
        id += "_HI" + dynamics + "_" #Hedging instruments
        for i in range(len(hedging_intruments_maturity)):
            if hedging_option_types[i] == False:
                id += 'C' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 
            else:
                id += 'P' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 

    #Simulate market
    #Simulate hedge portfolio
    print("-------- Portfolio simulation --------")
    portfolio_array, moneyness_array = hedged_portfolio(temporality, number_simulations, hedged_options_maturities, hedged_option_types, hedged_moneyness, hedged_positions, lower_bound, r, q, delta, S, B, tc, issmile)
            
    #Define number of features (S + time-to-temporality + B + H + [portfolio value, portfolios deltas, portfolios gammas] + present_hedging_options_prices) + moneyness_stats
    if len(hedging_intruments_maturity)!=0:
        num_features = 1 + 1 + (1)  #+ moneyness_array.shape[2] 
    else:
        num_features = 1 + 1 + (1)  #+ len(hedging_intruments_maturity) #+ moneyness_array.shape[2] 

    #Define time to temporality
    time_to_temp     = np.zeros(n_timesteps)
    time_to_temp[1:] = h      # [0,h,h,h,..,h]
    time_to_temp     = np.cumsum(time_to_temp) # [0,h,2h,...,Nh]
    time_to_temp     = time_to_temp[::-1]      # [Nh, (N-1)h,...,h,0]

    # Construct the train and test sets
    # The portfolio value V_{n} and the positions in hedging instruments will be added further into the code at each time-step
    features_set = np.zeros((n_timesteps, number_simulations, num_features))
    if len(hedging_intruments_maturity)!=0:
        option_price, deltas, gammas, TT, IV, new_option_price = hedging_instruments(temporality, hedging_intruments_maturity, hedging_option_types, hedging_option_moneyness, number_simulations, lower_bound, r, q, delta, S, B, tc, issmile, dynamics)
        #Delta gamma hedging
        lower_bound_gamma = 0.001
        strategy_g_0 = delta_gamma(portfolio_array, deltas, gammas, lower_bound_gamma, dynamics)
        features_set[:-1,:,:2] = strategy_g_0
    else:
        features_set[:,:,0] = np.transpose(portfolio_array[:,:,2])
    
    features_set[:,:,2] = np.transpose(portfolio_array[:,:,1])

    if len(hedging_intruments_maturity)!=0:
        if dynamics=='dynamic':
            H = np.zeros([n_timesteps,number_simulations,(n_timesteps-1),len(hedging_intruments_maturity)])
        elif dynamics == 'static':
            H = np.zeros([n_timesteps,number_simulations,2,len(hedging_intruments_maturity)])
        elif dynamics == "semistatic":
            H = np.zeros([n_timesteps,number_simulations,2,len(hedging_intruments_maturity)])


        if dynamics=='dynamic':
            H[:,:,:,:] = option_price
        elif dynamics == 'static':
            H[:,:,0,:] = option_price
            H[:,:,1,:] = 0
        elif dynamics == "semistatic":
            H[:,:,:,:] = option_price
    else:
        if dynamics=='dynamic':
            H = np.zeros([n_timesteps,number_simulations,(n_timesteps-1),1])
        elif dynamics == 'static':
            H = np.zeros([n_timesteps,number_simulations,2,1])
        elif dynamics == "semistatic":
            H = np.zeros([n_timesteps,number_simulations,2,1])

    #print("------------ Data scaling ------------")
    #Scaling data
    #for i in range(num_features):
    #    min = features_set[:,:,i].min()
    #    max = features_set[:,:,i].max()
    #    features_set[:,:,i] = (features_set[:,:,i]-min)/(max-min)

    #Split data into training set and validation set
    limit = int(4*(number_simulations*0.20))
    index_training = range(limit)
    index_test     = range(limit,limit+int(1*(number_simulations*0.20)))

    if trainphase==True:
        train_input      = features_set[:,index_training,:] 
        test_input       = features_set[:,index_test,:]
        HO_train         = H[:,index_training,:,:]
        HO_test          = H[:,index_test,:,:]
        cash_flows_train = np.transpose(portfolio_array[index_training,:,0])
        cash_flows_test  = np.transpose(portfolio_array[index_test,:,0])
        underlying_train = np.transpose(S[index_training,:])
        underlying_test  = np.transpose(S[index_test,:])
    else:
        train_input      = None
        test_input       = features_set
        HO_train         = None
        HO_test          = option_price
        cash_flows_train = None
        cash_flows_test  = np.transpose(portfolio_array[:,:,0])
        underlying_train = None
        underlying_test  = np.transpose(S)

    #Update factors for self-financing portfolio dynamics 
    risk_free_factor     = np.exp(r*h)   # exp(rh)
    dividendyield_factor = np.exp(q*h)   # exp(qh)
    print("-------- Simulation completed --------")

    return id, n_timesteps, train_input, test_input, risk_free_factor, dividendyield_factor, HO_train, HO_test, cash_flows_train, cash_flows_test, underlying_train, underlying_test


def load_standard_datasets(temporality):
    
    """Function that loads the sets to create the training set

        Parameters
        ----------
        maturity     : time to maturity of the options

        Returns
        -------
        S : Matrix of underlying asset prices
        B : Coefficients of the IV surface 
        H : Volatility of the underlying asset

      """
    
    # 1) Load datasets to train and test deep hedging algorithm (Simulated paths, Betas IV, volatilities)
    # 1.1) matrix of simulated stock prices
    test_period = range(500000)
    S = np.load(os.path.join(f"Stock_paths__random_f_{temporality}.npy"))[test_period]
    # 1.2) IV coefficients
    B = np.load(os.path.join(f"Betas_simulation__random_f_{temporality}.npy"))[test_period]
    # 1.3) Volatility
    H = np.load(os.path.join(f"H_simulation__random_f_{temporality}.npy"))[test_period]


    
    return S, B, H

def training_variables(temporality, hedged_options_maturities, hedged_option_types, hedged_moneyness, hedged_positions, hedging_intruments_maturity, hedging_option_types, hedging_option_moneyness, dynamics, lower_bound, r, q, delta, tc, issmile, backtest):

    """Function that loads the sets to create the training set

        Parameters
        ----------
        temporality                  : Temporality of the simulation
        r                            : Risk-free rate                  
        q                            : Dividend yield rate
        delta                        : Step-size (daily rebalancing)
        tc                           : Transaction cost level
        backtest                     : Simulation based on real or simulated data
        lower_bound                  : Lower bound to clip IV surface values
        issmile                      : To determine the type of greeks 
        hedged_options_maturities    : Option maturities for instruments to hedge
        hedged_option_types          : Option types {Call: False, Put: True}
        hedged_moneyness             : Option moneyness for instruments to hedge {"ATM","OTM","ITM"}
        hedged_positions             : Number of shares of each option to hedge
        hedging_intruments_maturity  : Option maturities for hedging instruments
        hedging_option_types         : Option types {Call: False, Put: True}
        hedging_option_moneyness     : Option moneyness for hedging instruments {"ATM","OTM","ITM"}
        dynamics                     : Simulation type {Single intrument: 'static', New instrument every day: 'dynamic'}     

        Returns
        -------
        id                   : Acronym for the hedging problem
        train_input          : Training set (normalized stock price and features)
        test_input           : Test set (normalized stock price and features)
        HO_train             : Prices of hedging instruments in the training set
        HO_test              : Prices of hedging instruments in the validation set
        cash_flows_train     : Cash flows of hedged portfolio for training set
        cash_flows_test      : Cash flows of hedged portfolio for validation set
        risk_free_factor     : Risk-free rate update factor exp(h*r)
        dividendyield_factor : Dividend yield update factor exp(h*d)
        underlying_train     : Underlying asset prices for training set
        underlying_test      : Underlying asset prices for validation set

      """

    owd = os.getcwd()
    try:
      #first change dir to build_dir path
      if backtest==True:
        trainphase = False
        os.chdir(os.path.join(main_folder, f"data/processed/Backtest/"))
      else:
        trainphase = True
        os.chdir(os.path.join(main_folder, f"data/processed/Training/"))

      S, B, H = load_standard_datasets(temporality)
      id, n_timesteps, train_input, test_input, risk_free_factor, dividendyield_factor, HO_train, HO_test, cash_flows_train, cash_flows_test, underlying_train, underlying_test = data_sets_preparation(temporality, S, B, H, hedged_options_maturities, hedged_option_types, hedged_moneyness, hedged_positions, hedging_intruments_maturity, hedging_option_types, hedging_option_moneyness, dynamics, lower_bound, r, q, delta, tc, issmile, trainphase)
      
    finally:
      #change dir back to original working directory (owd)
      os.chdir(owd)
      
    return id, n_timesteps, train_input, test_input, risk_free_factor, dividendyield_factor, HO_train, HO_test, cash_flows_train, cash_flows_test, underlying_train, underlying_test
  



