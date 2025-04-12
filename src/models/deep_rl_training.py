"""
Usage:
    1. cd src
    2. python3 models/deep_rl_training.py 
"""

import os, sys
from pathlib import Path
import warnings

# Set environment variable
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:urllib3'
# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

from src.utils import *
from src.data.full_data_loader import *
from src.models.deep_agent import train_network
from src.models.deep_agent import network_inference

def rl_agent(config_file_simulation,config_file_agent):
    
    """Function that trains the RL agent based on the configuration of the config files
    
    Parameters
    ----------
    config_file_simulation : simulation settings for the JIVR model and the underlying asset 
    config_file_agent : hyperparameters of the RL agent

    Output
    ----------
    deltas: hedging strategies
    
    """
    # 0) Default parameters 
    #Parameters of market simulation
    temporality = config_file_simulation["number_days"]     #Temporality of the simulation
    r = config_file_simulation['r']                         #Risk-free rate                  
    q = config_file_simulation['q']                         #Dividend yield rate
    delta = eval(config_file_simulation['size_step'])       #Step-size (daily rebalancing)
    tc_underlying = config_file_simulation['tc_underlying'] #Proportional transaction cost underlying
    tc_options = config_file_simulation['tc_options']       #Proportional transaction cost options
    tc = [tc_underlying/100,tc_options/100]                 #Transaction cost level
    backtest = False                                        #Simulation based on real or simulated data

    #Parameters of hedged portfolio
    hedged_options_maturities = config_file_agent['hedged_options_maturities']    #Option maturities for instruments to hedge
    hedged_option_types = config_file_agent['hedged_option_types']                #Option types {Call: False, Put: True}
    hedged_moneyness = config_file_agent['hedged_moneyness']                      #Option moneyness for instruments to hedge {"ATM","OTM","ITM"}
    hedged_positions = config_file_agent['hedged_positions']                      #Number of shares of each option to hedge

    #Parameters of hedging instruments
    hedging_intruments_maturity = config_file_agent['hedging_intruments_maturity']     #Option maturities for hedging instruments
    hedging_option_types = config_file_agent['hedging_option_types']                   #Option types {Call: False, Put: True}
    hedging_option_moneyness = config_file_agent['hedging_option_moneyness']           #Option moneyness for hedging instruments {"ATM","OTM","ITM"}
    dynamics = config_file_agent['dynamics']                                           #Simulation type {Single intrument: 'static', New instrument every day: 'dynamic'}

    #Parameters for IV and greeks simulation
    lower_bound = config_file_simulation['vol_lower_bound']  #Lower bound for IV
    issmile = 'BS' #Simulate deltas and gammas under Black-Scholes formula

    # 1) Loading data in the right shape for RL-agent input
    id, n_timesteps, train_input, test_input, risk_free_factor, dividendyield_factor, HO_train, HO_test, cash_flows_train, cash_flows_test, underlying_train, underlying_test = training_variables(temporality, hedged_options_maturities, hedged_option_types, hedged_moneyness, hedged_positions, hedging_intruments_maturity, hedging_option_types, hedging_option_moneyness, dynamics, lower_bound, r, q, delta, tc, issmile, backtest)

    # 2) First layer of RL agent hyperparameters
    # First layer of parameters
    network           = config_file_agent['network']                       # Neural network architecture {"LSTM","RNNFNN","FFNN"}
    nbs_point_traj    = n_timesteps                                        # time steps 
    batch_size        = config_file_agent['batch_size']                    # batch size {296,1000}
    nbs_input         = train_input.shape[2]                               # number of features
    nbs_units         = config_file_agent['nbs_units']                     # neurons per layer/cell
    hedging_options   = len(hedging_intruments_maturity)                   # Hedging options counter
    nbs_assets        = 2 if hedging_options==0 else (hedging_options+1)   # number of hedging intruments (this inlcudes the underlying asset)
    constraint_min    = 1                     # Proportion of initial portfolio value for the tracking error threshold 
    lambda_m          = 1                     # lambda for lagrange multiplier (soft constraint)
    autoencoder       = False                 # autoencoder component included
    nbs_latent_spc    = 8                     # dimension of the latent space
    lambda_auto       = 1                     # regularization parameter to ponderate autoencoder loss function
    loss_type         = config_file_agent['loss_type']         # loss function {"CVaR","MSE","SMSE"}
    lr                = config_file_agent['lr']                # learning rate of the Adam optimizer
    dropout_par       = config_file_agent['dropout_par']       # dropout regularization parameter 

    # 2) Second layer of RL agent hyperparameters
    transaction_cost  = tc                                  # Transaction cost levels
    rebalancing_const = [1000.0, 0.1]                       # Rebalancing constraints hyperparameters sharpness and epsilon
    riskaversion      = config_file_agent['riskaversion']   # CVaR confidence level (0,1)
    epochs            = config_file_agent['epochs']         # Number of epochs, training iterations

    # 3) Third layer of parameters
    id                   = id                   # Acronym for the hedging problem
    train_input          = train_input          # Training set (normalized stock price and features)
    test_input           = test_input           # Test set (normalized stock price and features)
    HO_train             = HO_train             # Prices of hedging instruments in the training set
    HO_test              = HO_test              # Prices of hedging instruments in the validation set
    cash_flows_train     = cash_flows_train     # Cash flows of hedged portfolio for training set
    cash_flows_test      = cash_flows_test      # Cash flows of hedged portfolio for validation set
    risk_free_factor     = risk_free_factor     # Risk-free rate update factor exp(h*r)
    dividendyield_factor = dividendyield_factor # Dividend yield update factor exp(h*d)
    underlying_train     = underlying_train     # Underlying asset prices for training set
    underlying_test      = underlying_test      # Underlying asset prices for validation set
            
    # 4) Fourth layer of parameters
    display_plot    = config_file_agent['display_plot']  # Display plot of training and validation loss 

    # 5) Train RL agent
    loss_train_epoch = train_network(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lambda_m, constraint_min,
                                            loss_type, lr, dropout_par, train_input, underlying_train, underlying_test, HO_train, HO_test,
                                            cash_flows_train, cash_flows_test, risk_free_factor,dividendyield_factor, transaction_cost,
                                                riskaversion, test_input, epochs, display_plot, id, autoencoder, nbs_latent_spc, lambda_auto, rebalancing_const)
    print("--- Deep agent trained and stored in ../models/.. ---")

    # 6) Compute hedging strategy
    deltas,latent, portfolio, constraint, name = network_inference(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets,
                                                            lambda_m, constraint_min,loss_type, lr, dropout_par, train_input,
                                                            underlying_train, underlying_test, HO_train, HO_test, cash_flows_train,
                                                                cash_flows_test, risk_free_factor, dividendyield_factor, transaction_cost, 
                                                                riskaversion, test_input, epochs, display_plot, id, autoencoder,
                                                                nbs_latent_spc, lambda_auto, rebalancing_const)
    
    print("--- Hedging startegy stored in ../results/Training/.. ---")

    return deltas
    
if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file = load_config(os.path.join(main_folder,'cfgs','config_agent.yml'))
    config_file_agent = config_file["agent"]
    config_file = load_config(os.path.join(main_folder,'cfgs','config_simulation.yml'))
    config_file_simulation = config_file["simulation"]
    _ = rl_agent(config_file_simulation,config_file_agent)