import logging
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def load_config(config_file):
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config

def moneyness_def(S, moneyness, isput):
  
    """Function to compute strike price based on the option moneyness 

        Parameters
        ----------
        S                        : numpy array with underlying asset price simulation
        moneyness                : Moneyness of the hedging instrument regardless the dynamics ATM, OTM, ITM
        isput                    : boolean variable to determine european option type

        Returns
        -------
        K                        : numpy array with strike price

    """

    moneyness_list = ["ATM","ITM","OTM"]
    differences    = [0,-10,10]
    idx         = moneyness_list.index(moneyness)
    # 1.2) Moneyness and strike
    if isput == True:
     if moneyness_list[idx]=="ITM":
       K = S + differences[idx+1]
     elif moneyness_list[idx]=="OTM":
       K = S + differences[idx-1]
     else:
       K = S + differences[idx]
    else:
       K = S + differences[idx]
    
    return K

def name_strategy(hedged_option_types, hedged_options_maturities, hedged_moneyness, hedging_option_types, hedged_positions, hedging_intruments_maturity, hedging_option_moneyness, tc, rebalancing_const, dynamics, loss_type):

    transaction_cost_name = f"TC_{str('_'.join(map(str, [x * 100 for x in tc])))}" if sum(tc)!= 0 else str(0)
    rebalance_name = f"Rebalance_{str('_'.join(map(str, [x * 100 for x in rebalancing_const])))}" if sum(tc)!= 0 else str(0)

    #ID hedged instruments
    id = "HO_" #Hedged options
    for i in range(len(hedged_options_maturities)):
        if hedged_option_types[i] == False:
            id += 'C' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 
        else:
            id += 'P' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 

    #ID hedigng instruments
    id += "_HI" + dynamics + "_" #Hedging instruments
    for i in range(len(hedging_intruments_maturity)):
        if hedging_option_types[i] == False:
            id += 'C' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 
        else:
            id += 'P' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 

    if loss_type == "CVaR":
        name = f"RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{str(int(95*100))}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"
    else:
        name = f"RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"

    return name

def name_strategy_l(hedged_option_types, hedged_options_maturities, hedged_moneyness, hedging_option_types, hedged_positions, hedging_intruments_maturity, hedging_option_moneyness, tc, rebalancing_const, dynamics, loss_type):

    transaction_cost_name = f"TC_{str('_'.join(map(str, [x * 100 for x in tc])))}" if sum(tc)!= 0 else str(0)
    rebalance_name = f"learned_reebalance_integral"

    #ID hedged instruments
    id = "HO_" #Hedged options
    for i in range(len(hedged_options_maturities)):
        if hedged_option_types[i] == False:
            id += 'C' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 
        else:
            id += 'P' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 

    #ID hedigng instruments
    id += "_HI" + dynamics + "_" #Hedging instruments
    for i in range(len(hedging_intruments_maturity)):
        if hedging_option_types[i] == False:
            id += 'C' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 
        else:
            id += 'P' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 

    if loss_type == "CVaR":
        name = f"RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{str(int(95))}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"
    else:
        name = f"RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"

    return name

def batch_objectivefunction(hedging_err_lim, loss_function, batch_size, idx):
    sample_size = hedging_err_lim.shape[0]
    #idx         = np.arange(sample_size)
    results_batch = np.zeros([int(sample_size/batch_size)])
    for i in range(int(sample_size/batch_size)):
        indices = idx[i*batch_size : (i+1)*batch_size]
        hedging_err = hedging_err_lim[indices]
        if loss_function=="CVaR":
            results_batch[i] = np.mean(np.sort(hedging_err)[int(0.95*hedging_err.shape[0]):])
        elif loss_function=="MSE":
            results_batch[i] = np.mean(np.square(hedging_err))
        elif loss_function=="SMSE":
            results_batch[i] = np.mean(np.square(np.where(hedging_err>0,hedging_err,0)))

    return results_batch

def name_strategy_l_newband(hedged_option_types, hedged_options_maturities, hedged_moneyness, hedging_option_types, hedged_positions, hedging_intruments_maturity, hedging_option_moneyness, tc, rebalancing_const, dynamics, loss_type):

    transaction_cost_name = f"TC_{str('_'.join(map(str, [x * 100 for x in tc])))}" if sum(tc)!= 0 else str(0)
    rebalance_name = f"learned_reebalance_integral"

    #ID hedged instruments
    id = "HO_" #Hedged options
    for i in range(len(hedged_options_maturities)):
        if hedged_option_types[i] == False:
            id += 'C' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 
        else:
            id += 'P' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 

    #ID hedigng instruments
    id += "_HI" + dynamics + "_" #Hedging instruments
    for i in range(len(hedging_intruments_maturity)):
        if hedging_option_types[i] == False:
            id += 'C' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 
        else:
            id += 'P' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 

    if loss_type == "CVaR":
        name = f"C_RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{str(int(95))}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"
    else:
        name = f"C_RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"

    return name

def name_strategy_l_newband_2(hedged_option_types, hedged_options_maturities, hedged_moneyness, hedging_option_types, hedged_positions, hedging_intruments_maturity, hedging_option_moneyness, tc, rebalancing_const, dynamics, loss_type):

    transaction_cost_name = f"TC_{str('_'.join(map(str, [x * 100 for x in tc])))}" if sum(tc)!= 0 else str(0)
    rebalance_name = f"learned_reebalance_integral"

    #ID hedged instruments
    id = "HO_" #Hedged options
    for i in range(len(hedged_options_maturities)):
        if hedged_option_types[i] == False:
            id += 'C' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 
        else:
            id += 'P' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 

    #ID hedigng instruments
    id += "_HI" + dynamics + "_" #Hedging instruments
    for i in range(len(hedging_intruments_maturity)):
        if hedging_option_types[i] == False:
            id += 'C' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 
        else:
            id += 'P' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 

    if loss_type == "CVaR":
        name = f"C2_RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{str(int(95))}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"
    else:
        name = f"C2_RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"

    return name

def name_strategy_l_newband_3(hedged_option_types, hedged_options_maturities, hedged_moneyness, hedging_option_types, hedged_positions, hedging_intruments_maturity, hedging_option_moneyness, tc, rebalancing_const, dynamics, loss_type):

    transaction_cost_name = f"TC_{str('_'.join(map(str, [x * 100 for x in tc])))}" if sum(tc)!= 0 else str(0)
    rebalance_name = f"learned_reebalance_integral"

    #ID hedged instruments
    id = "HO_" #Hedged options
    for i in range(len(hedged_options_maturities)):
        if hedged_option_types[i] == False:
            id += 'C' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 
        else:
            id += 'P' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 

    #ID hedigng instruments
    id += "_HI" + dynamics + "_" #Hedging instruments
    for i in range(len(hedging_intruments_maturity)):
        if hedging_option_types[i] == False:
            id += 'C' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 
        else:
            id += 'P' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 

    if loss_type == "CVaR":
        name = f"C4_RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{str(int(95))}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"
    else:
        name = f"C4_RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"

    return name


def name_strategy_soft_constraint(hedged_option_types, hedged_options_maturities, hedged_moneyness, hedging_option_types, hedged_positions, hedging_intruments_maturity, hedging_option_moneyness, tc, rebalancing_const, dynamics, loss_type, lambda_val):

    transaction_cost_name = f"TC_{str('_'.join(map(str, [x * 100 for x in tc])))}" if sum(tc)!= 0 else str(0)
    rebalance_name = f"learned_reebalance_integral"

    #ID hedged instruments
    id = "HO_" #Hedged options
    for i in range(len(hedged_options_maturities)):
        if hedged_option_types[i] == False:
            id += 'C' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 
        else:
            id += 'P' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 

    #ID hedigng instruments
    id += "_HI" + dynamics + "_" #Hedging instruments
    for i in range(len(hedging_intruments_maturity)):
        if hedging_option_types[i] == False:
            id += 'C' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 
        else:
            id += 'P' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 

    val_ = str(int(lambda_val)) if ((lambda_val==1) or (lambda_val==0) or (lambda_val==2)) else str(lambda_val)

    if loss_type == "CVaR":
        name = f"C2_RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{str(int(95))}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{val_}.npy"
    else:
        name = f"C2_RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{val_}.npy"

    return name


def name_strategy_l_newband_2_notthresh(hedged_option_types, hedged_options_maturities, hedged_moneyness, hedging_option_types, hedged_positions, hedging_intruments_maturity, hedging_option_moneyness, tc, rebalancing_const, dynamics, loss_type):

    transaction_cost_name = f"TC_{str('_'.join(map(str, [x * 100 for x in tc])))}" if sum(tc)!= 0 else str(0)
    rebalance_name = f"learned_reebalance_integral"

    #ID hedged instruments
    id = "HO_" #Hedged options
    for i in range(len(hedged_options_maturities)):
        if hedged_option_types[i] == False:
            id += 'C' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 
        else:
            id += 'P' + str(hedged_options_maturities[i]) + hedged_moneyness[i] + 'pos' + str(hedged_positions[i]) 

    #ID hedigng instruments
    id += "_HI" + dynamics + "_" #Hedging instruments
    for i in range(len(hedging_intruments_maturity)):
        if hedging_option_types[i] == False:
            id += 'C' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 
        else:
            id += 'P' + str(hedging_intruments_maturity[i]) + hedging_option_moneyness[i] 

    if loss_type == "CVaR":
        name = f"No_thresh_C2_RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{str(int(95))}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"
    else:
        name = f"No_thresh_C2_RNNFNN_noauto_dropout_{str(int(0.5*100))}_{loss_type}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{1}.npy"

    return name