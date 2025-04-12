"""
Usage:
    1. cd src
    2. python models/matchmaking/advisor_to_company_match.py
"""


import os, sys
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap


class hedging_evaluation(object):
    
    """Class to implement the delta-hedging framework
    
    Parameters
    ----------
    Time_steps : int, optional
    number_sumlations: int, optional
        
    """
    
    def __init__(self, dynamic=False, transaction_cost = [0,0], r = 0.026623194, q = 0.01772245):

        # Parameters
        self.dynamics = dynamic                       #Simulation type {Single intrument: 'static', New instrument every day: 'dynamic'}
        self.r = r                                    #Risk-free rate
        self.q = q                                    #Dividend yield rate
        self.transaction_cost = transaction_cost      #Transaction cost level

    def hedging_error_vector(self, portfolio_value, cash_flows, underlying_asset, hedging_instruments, positions, close_limit_days, hedging_intruments_maturity, rebalancing_const):
        
        #General values
        time_steps = underlying_asset.shape[1]-1
        number_simulations = underlying_asset.shape[0]
        n_hedging_instruments = positions.shape[2]
        limit = time_steps-close_limit_days
        
        #Hedging portfolio
        hedging_portfolio = np.zeros([number_simulations,(time_steps+1)])
        hedging_error = np.zeros([number_simulations,(time_steps+1)])
        bank_account = np.zeros([number_simulations,(time_steps)])
        cost_matrix = np.zeros([number_simulations,(time_steps)])
        
        #Compute P&L for the hedging startegies
        hedging_portfolio[:,0] = cash_flows[:,0]
        if self.dynamics == 'static':
            #Homologate arrays dimensions 
            positions = np.transpose(positions, axes=(1, 0, 2))
            hedging_instruments = np.transpose(hedging_instruments, axes=(1, 0, 2))
            #Computing P&L of hedging portfolio
            for time_step in range(time_steps):
                #Transacition cost computation for the first time-step
                if time_step == 0:
                    cost = np.abs(positions[:,time_step,0]*underlying_asset[:,time_step])*self.transaction_cost[0]
                    for a in range(n_hedging_instruments-1):
                        cost += np.abs(positions[:,time_step,a+1]*hedging_instruments[:,time_step,a])*self.transaction_cost[1]
                    cost_matrix[:,0] = cost
                else:
                    #Compute transaction cost when rebalancing 
                    cost =  np.abs(underlying_asset[:,time_step]*(positions[:,time_step,0]-positions[:,time_step-1,0]))*self.transaction_cost[0]
                    for a in range(n_hedging_instruments-1):
                        cost +=  np.abs(hedging_instruments[:,time_step,a]*(positions[:,time_step,a+1]-positions[:,time_step-1,a+1]))*self.transaction_cost[1]
                    cost_matrix[:,time_step] = cost_matrix[:,time_step-1] + cost*np.exp((-1)*self.r*(time_step/252))

                #Compute bank account values based on rebalancing dynamics
                bank_account[:,time_step] = hedging_portfolio[:,time_step] - positions[:,time_step,0]*underlying_asset[:,time_step] - cost
                for a in range(n_hedging_instruments-1):
                    bank_account[:,time_step] += -1*positions[:,time_step,a+1]*hedging_instruments[:,time_step,a]

                #New portfolio value
                hedging_portfolio[:,(time_step+1)] = bank_account[:,time_step]*np.exp(self.r/252) + positions[:,time_step,0]*underlying_asset[:,(time_step+1)]*np.exp(self.q/252)
                for a in range(n_hedging_instruments-1):
                    hedging_portfolio[:,(time_step+1)] += positions[:,time_step,a+1]*hedging_instruments[:,time_step+1,a]
                hedging_portfolio[:,(time_step+1)] += cash_flows[:,(time_step+1)]
            
            # Substract cost for liquidate the position at the end of the hedging period
            cost = np.zeros([number_simulations])
            for a in range(n_hedging_instruments-1):
                cost += np.abs(positions[:,time_step,a+1]*hedging_instruments[:,time_step+1,a])*self.transaction_cost[1]
            hedging_portfolio[:,(time_step+1)] = hedging_portfolio[:,(time_step+1)] - cost
            cost_matrix[:,time_step] = cost_matrix[:,time_step] + cost*np.exp((-1)*self.r*((time_step+1)/252))

            #Hedging error computation
            hedging_error[:,:-1] = -1*(hedging_portfolio[:,:-1]-portfolio_value[:,:-1])
            hedging_error[:,time_steps] = -1*hedging_portfolio[:,time_steps]
            hedging_error_limit = hedging_error[:,limit]
            cost_limit = cost_matrix[:,limit-1]

        elif self.dynamics == 'dynamic':

            #Computing P&L of hedging portfolio
            for time_step in range(time_steps):
                #Transacition cost computation for the first time-step
                if time_step == 0:
                    cost = np.abs(positions[time_step,:,0]*underlying_asset[:,time_step])*self.transaction_cost[0]
                    for a in range(n_hedging_instruments-1):
                        cost += np.abs(positions[time_step,:,a+1]*hedging_instruments[time_step,:,time_step,a])*self.transaction_cost[1]
                    cost_matrix[:,0] = cost
                else:
                    #Compute transaction cost when rebalancing 
                    cost =  np.abs(underlying_asset[:,time_step]*(positions[time_step,:,0]-positions[time_step-1,:,0]))*self.transaction_cost[0]
                    for a in range(n_hedging_instruments-1):
                        cost += np.abs(positions[time_step,:,a+1]*hedging_instruments[time_step,:,time_step,a])*self.transaction_cost[1]
                    cost_matrix[:,time_step] = cost_matrix[:,time_step-1] + cost*np.exp((-1)*self.r*(time_step/252))

                #Compute bank account values based on rebalancing dynamics
                cumulative_combination = np.zeros([number_simulations])
                for a in range(n_hedging_instruments-1):
                    for cum_t in range(time_step+1):
                        cumulative_combination += positions[cum_t,:,a+1]*hedging_instruments[time_step,:,cum_t,a]*((time_step-cum_t)!=hedging_intruments_maturity[a])
                
                bank_account[:,time_step] = hedging_portfolio[:,time_step] - positions[time_step,:,0]*underlying_asset[:,time_step] - cumulative_combination - cost

                #New portfolio value
                cumulative_combination = np.zeros([number_simulations])
                for a in range(n_hedging_instruments-1):
                    for cum_t in range(time_step+1):
                        cumulative_combination += positions[cum_t,:,a+1]*hedging_instruments[time_step+1,:,cum_t,a]
                
                hedging_portfolio[:,(time_step+1)] = bank_account[:,time_step]*np.exp(self.r/252) + positions[time_step,:,0]*underlying_asset[:,(time_step+1)]*np.exp(self.q/252) + cumulative_combination

                hedging_portfolio[:,(time_step+1)] += cash_flows[:,(time_step+1)]
                

            #Hedging error computation
            hedging_error[:,:-1] = -1*(hedging_portfolio[:,:-1]-portfolio_value[:,:-1])
            hedging_error[:,time_steps] = -1*hedging_portfolio[:,time_steps]
            hedging_error_limit = hedging_error[:,limit]
            cost_limit = cost_matrix[:,limit-1]

        elif self.dynamics == "semistatic":
            #Homologate arrays dimensions 
            positions = np.transpose(positions, axes=(1, 0, 2))
            hedging_instruments = np.transpose(hedging_instruments, axes=(1, 0, 2, 3))
            #Computing P&L of hedging portfolio
            for time_step in range(time_steps):
                #Transacition cost computation for the first time-step
                if time_step == 0:
                    cost = np.abs(positions[:,time_step,0]*underlying_asset[:,time_step])*self.transaction_cost[0]
                    for a in range(n_hedging_instruments-1):
                        cost += np.abs(positions[:,time_step,a+1]*hedging_instruments[:,time_step,0,a])*self.transaction_cost[1]
                    cost_matrix[:,0] = cost
                else:
                    #Compute transaction cost when rebalancing 
                    cost =  np.abs(underlying_asset[:,time_step]*(positions[:,time_step,0]-positions[:,time_step-1,0]))*self.transaction_cost[0]
                    for a in range(n_hedging_instruments-1):
                        indicator = (hedging_instruments[:,time_step,1,a]>0)
                        cost +=  np.abs(hedging_instruments[:,time_step,0,a]*(positions[:,time_step,a+1]-positions[:,time_step-1,a+1]))*(1-indicator)*self.transaction_cost[1] + (np.abs(hedging_instruments[:,time_step,1,a]*(positions[:,time_step-1,a+1]))+np.abs(hedging_instruments[:,time_step,0,a]*(positions[:,time_step,a+1])))*(indicator)*self.transaction_cost[1]
                    cost_matrix[:,time_step] = cost_matrix[:,time_step-1] + cost*np.exp((-1)*self.r*(time_step/252))

                #Compute bank account values based on rebalancing dynamics
                bank_account[:,time_step] = hedging_portfolio[:,time_step] - positions[:,time_step,0]*underlying_asset[:,time_step] - cost 
                for a in range(n_hedging_instruments-1):
                    bank_account[:,time_step] += -1*positions[:,time_step,a+1]*hedging_instruments[:,time_step,0,a]

                #New portfolio value
                hedging_portfolio[:,(time_step+1)] = bank_account[:,time_step]*np.exp(self.r/252) + positions[:,time_step,0]*underlying_asset[:,(time_step+1)]*np.exp(self.q/252)
                for a in range(n_hedging_instruments-1):
                    #becareful
                    indicator = (hedging_instruments[:,time_step+1,1,a]>0) 
                    hedging_portfolio[:,(time_step+1)] += positions[:,time_step,a+1]*(hedging_instruments[:,time_step+1,0,a]*(1-indicator)+hedging_instruments[:,time_step+1,1,a]*(indicator))
                hedging_portfolio[:,(time_step+1)] += cash_flows[:,(time_step+1)]

            # Substract cost for liquidate the position at the end of the hedging period
            cost = np.zeros([number_simulations])
            for a in range(n_hedging_instruments-1):
                cost += np.abs(positions[:,time_step,a+1]*hedging_instruments[:,time_step+1,0,a])*self.transaction_cost[1]
            hedging_portfolio[:,(time_step+1)] = hedging_portfolio[:,(time_step+1)] - cost
            #Include the last cost for closing the postion
            cost_matrix[:,time_step] = cost_matrix[:,time_step] + cost*np.exp((-1)*self.r*((time_step+1)/252))
            
            #Hedging error computation
            hedging_error[:,:-1] = -1*(hedging_portfolio[:,:-1]-portfolio_value[:,:-1])
            hedging_error[:,time_steps] = -1*hedging_portfolio[:,time_steps]
            hedging_error_limit = hedging_error[:,limit]
            cost_limit = cost_matrix[:,limit-1]

        elif self.dynamics == "semistatic_delta_gamma":

            #Homologate arrays dimensions 
            positions = np.transpose(positions, axes=(1, 0, 2))
            hedging_instruments = np.transpose(hedging_instruments, axes=(1, 0, 2, 3))
            #Computing P&L of hedging portfolio
            for time_step in range(time_steps):
                #Transacition cost computation for the first time-step
                if time_step == 0:
                    cost = np.abs(positions[:,time_step,0]*underlying_asset[:,time_step])*self.transaction_cost[0]
                    for a in range(n_hedging_instruments-1):
                        cost += np.abs(positions[:,time_step,a+1]*hedging_instruments[:,time_step,0,a])*self.transaction_cost[1]
                    cost_matrix[:,0] = cost
                else:
                    # Desition for rebalancing [batch_size,self.nbs_assets]
                    if np.sum(self.transaction_cost)!=0:
                        indicator_1 = np.concatenate(((underlying_asset[:,time_step]*0)[:,np.newaxis], hedging_instruments[:,time_step,1,:]), axis=-1)>0
                        indicator_2 = (np.abs(positions[:,time_step,:] - positions[:,time_step-1,:])*np.concatenate(((underlying_asset[:,time_step]*self.transaction_cost[0])[:,np.newaxis], hedging_instruments[:,time_step,0,:]*self.transaction_cost[1]), axis=-1))>= rebalancing_const
                        indicador_3 = (indicator_1 + indicator_2)>0
                        positions[:,time_step,:] = positions[:,time_step,:]*indicador_3 + positions[:,time_step-1,:]*(1-indicador_3)

                    #Compute transaction cost when rebalancing 
                    cost =  np.abs(underlying_asset[:,time_step]*(positions[:,time_step,0]-positions[:,time_step-1,0]))*self.transaction_cost[0]
                    for a in range(n_hedging_instruments-1):
                        indicator = (hedging_instruments[:,time_step,1,a]>0)
                        cost +=  np.abs(hedging_instruments[:,time_step,0,a]*(positions[:,time_step,a+1]-positions[:,time_step-1,a+1]))*(1-indicator)*self.transaction_cost[1] + (np.abs(hedging_instruments[:,time_step,1,a]*(positions[:,time_step-1,a+1]))+np.abs(hedging_instruments[:,time_step,0,a]*(positions[:,time_step,a+1])))*(indicator)*self.transaction_cost[1]
                    cost_matrix[:,time_step] = cost_matrix[:,time_step-1] + cost*np.exp((-1)*self.r*(time_step/252))

                #Compute bank account values based on rebalancing dynamics
                bank_account[:,time_step] = hedging_portfolio[:,time_step] - positions[:,time_step,0]*underlying_asset[:,time_step] - cost 
                for a in range(n_hedging_instruments-1):
                    bank_account[:,time_step] += -1*positions[:,time_step,a+1]*hedging_instruments[:,time_step,0,a]

                #New portfolio value
                hedging_portfolio[:,(time_step+1)] = bank_account[:,time_step]*np.exp(self.r/252) + positions[:,time_step,0]*underlying_asset[:,(time_step+1)]*np.exp(self.q/252)
                for a in range(n_hedging_instruments-1):
                    #becareful
                    indicator = (hedging_instruments[:,time_step+1,1,a]>0) 
                    hedging_portfolio[:,(time_step+1)] += positions[:,time_step,a+1]*(hedging_instruments[:,time_step+1,0,a]*(1-indicator)+hedging_instruments[:,time_step+1,1,a]*(indicator))
                hedging_portfolio[:,(time_step+1)] += cash_flows[:,(time_step+1)]

            # Substract cost for liquidate the position at the end of the hedging period
            cost = np.zeros([number_simulations])
            for a in range(n_hedging_instruments-1):
                cost += np.abs(positions[:,time_step,a+1]*hedging_instruments[:,time_step+1,0,a])*self.transaction_cost[1]
            hedging_portfolio[:,(time_step+1)] = hedging_portfolio[:,(time_step+1)] - cost
            #Include the last cost for closing the postion
            cost_matrix[:,time_step] = cost_matrix[:,time_step] + cost*np.exp((-1)*self.r*((time_step+1)/252))
            
            #Hedging error computation
            hedging_error[:,:-1] = -1*(hedging_portfolio[:,:-1]-portfolio_value[:,:-1])
            hedging_error[:,time_steps] = -1*hedging_portfolio[:,time_steps]
            hedging_error_limit = hedging_error[:,limit]
            cost_limit = cost_matrix[:,limit-1]

        return hedging_portfolio, hedging_error, hedging_error_limit, cost_matrix, bank_account, np.transpose(positions, axes=(1, 0, 2))
    
def delta_gamma(portfolio_array, deltas, gammas, lower_bound_gamma, dynamics):

    """Function to compute delta-gamma hedging positions
    
    Parameters
    ----------
    portfolio_array    : array with deltas/gammas of hedged portfolio
    deltas             : hedging instruments deltas
    gammas             : hedging instruments gammas
    lower_bound_gamma  : lower bound to clip numerator for gamma positions
    dynamics           : define market hedigng instruments dynamics
        
    """

    strategy = np.zeros([gammas.shape[0]-1,gammas.shape[1],2])
    deltas_p = portfolio_array[:,:,2]
    gammas_p = portfolio_array[:,:,3]

    #Compute delta-gamma positions for static scenario
    if ((dynamics == 'static') or (dynamics == "semistatic")):

        deltas_hi = np.transpose(deltas, axes=(1, 0, 2))[:,:,0]
        gammas_hi = np.transpose(gammas, axes=(1, 0, 2))[:,:,0]

        position_hedging_option = (gammas_p[:,:-1] / np.maximum(gammas_hi[:,:-1],lower_bound_gamma))
        position_underlying_asset = deltas_p[:,:-1] - deltas_hi[:,:-1]*position_hedging_option

        strategy[:,:,0] = np.transpose(position_underlying_asset)
        strategy[:,:,1] = np.transpose(position_hedging_option)

    #Compute delta-gamma positions for dynamic scenario
    else:
        for time_step in range(gammas.shape[0]-1):
            cumulant_gamma = 0
            for i in range(time_step):
                cumulant_gamma += gammas[time_step,:,i,0]*strategy[i,:,1]
            strategy[time_step,:,1] = (gammas_p[:,time_step]-cumulant_gamma)/np.maximum(gammas[time_step,:,time_step,0],lower_bound_gamma)

            cumulant_delta = 0
            for i in range(time_step+1):
                cumulant_delta += deltas[time_step,:,i,0]*strategy[i,:,1]
            strategy[time_step,:,0] = deltas_p[:,time_step] - cumulant_delta

    return strategy
    
def statistics(hedging_err,hedging_portfolio,cost_limit,strategy,hedging_error):

    paths_proportion = ((np.abs(np.diff(strategy, axis=0))>0).sum(axis=0)/(strategy.shape[0]-1)).mean(axis=0)*100

    "Mean P&L"
    loss = np.mean(-1*hedging_err)
    #"Delta - CVaR"
    #loss = np.append(loss,np.mean(np.sort(hedging_err-np.mean(hedging_err))[int(0.95*hedging_err.shape[0]):]))
    "CVaR - 95"
    loss = np.append(loss,np.mean(np.sort(hedging_err)[int(0.95*hedging_err.shape[0]):]))
    "MSE"
    loss = np.append(loss,np.mean(np.square(hedging_err)))
    "SMSE"
    loss = np.append(loss,np.mean(np.square(np.where(hedging_err>0,hedging_err,0))))
    #"Profitability-ratio"
    #loss = np.append(loss,np.mean(-1*hedging_err)/np.mean(np.sort(hedging_err-np.mean(hedging_err))[int(0.95*hedging_err.shape[0]):]))
    #"Paths with negative values"
    #loss = np.append(loss,((hedging_portfolio<0).sum(axis=1)>0).sum()/hedging_err.shape[0])
    "Average cost"
    loss = np.append(loss,cost_limit[:,-1].mean())
    "Proportion of rebalancing underlying and option"
    loss = np.append(loss,paths_proportion[0])
    loss = np.append(loss,paths_proportion[1])
    #Proportin of paths reaching the tracking error threshold
    loss = np.append(loss,(((hedging_error[:,:-2]>hedging_portfolio[:,[0]]).sum(axis=1)>0).sum()/100000)*100)

    stats_1 = pd.DataFrame(loss)
    stats_1.index = ["Avg P&L","CVaR","MSE","SMSE","Avg Cost","Rebalancing Underlying", "Rebalancing Option", "Proportion tracking error %"]
    stats_1 = stats_1.T

    return stats_1

def statistics_zero(hedging_err,hedging_portfolio,cost_limit,strategy,hedging_error):

    "Mean P&L"
    loss = np.mean(-1*hedging_err)
    #"Standard deviation P&L"
    loss = np.append(loss,np.std(-1*hedging_err))
    #"Delta - CVaR"
    #loss = np.append(loss,np.mean(np.sort(hedging_err-np.mean(hedging_err))[int(0.95*hedging_err.shape[0]):]))
    "CVaR - 95"
    loss = np.append(loss,np.mean(np.sort(hedging_err)[int(0.95*hedging_err.shape[0]):]))
    "MSE"
    loss = np.append(loss,np.mean(np.square(hedging_err)))
    "SMSE"
    loss = np.append(loss,np.mean(np.square(np.where(hedging_err>0,hedging_err,0))))
    #"Profitability-ratio"
    loss = np.append(loss,np.mean(-1*hedging_err)/np.std(-1*hedging_err))
    #"Paths with negative values"
    #loss = np.append(loss,((hedging_portfolio<0).sum(axis=1)>0).sum()/hedging_err.shape[0])
    "Average cost"
    #loss = np.append(loss,cost_limit[:,-1].mean())
    "Proportion of rebalancing underlying and option"
    #loss = np.append(loss,paths_proportion[0])
    #loss = np.append(loss,paths_proportion[1])
    #Proportin of paths reaching the tracking error threshold
    #loss = np.append(loss,((hedging_error[:,:-1]>hedging_portfolio[0,:-1]).sum(axis=1)>0).sum()/10000)

    stats_1 = pd.DataFrame(loss)
    stats_1.index = ["Avg P&L","Std P&L","CVaR","MSE","SMSE","Sharpe Ratio"]
    stats_1 = stats_1.T

    return stats_1

def empirical_copula(X,Y):
    comparison_X = (X <= X[:, np.newaxis])
    comparison_Y = (Y <= Y[:, np.newaxis])
    preliminar_X = comparison_X.sum(axis=1)
    preliminar_Y = comparison_Y.sum(axis=1)

    n = len(X)
    sw = 0
    for i in range(n):
        sw += np.abs((comparison_X[i]*comparison_Y).sum(axis=1)/n - (preliminar_X[i]*preliminar_Y)/(n**2)).sum()
    sw = (12*sw)/(n**2-1)

    return sw

def network_evaluation(name, test_input, latent, deltas, portfolio, autoencoder):

    #Loss function metrics
    loss = np.load(os.path.join(main_folder, f"data/results/Training/results_extra/loss_functions/loss_{name}.npy"))
    loss_metrics = loss[np.where(loss[:,1]==loss[:,1].min())[0][0],[1,3,5]]
    loss_metrics = pd.DataFrame(loss_metrics).T
    loss_metrics.columns = ["Loss_function","Loss_autoencoder","Soft_constraint"]
    loss_metrics.to_csv(os.path.join(main_folder, f"data/results/Training/results_extra/loss_functions/loss_{name}.csv"))

    new_input  = test_input[:-1,:,:]
    new_input  = new_input.reshape(new_input.shape[0] * new_input.shape[1], new_input.shape[2])

    if autoencoder == True:

        new_latent = latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])

        #One lag for deltas
        deltas[1:,:,:] = deltas[:-1,:,:] 
        deltas[0,:,:] = 0
        deltas = deltas.reshape(deltas.shape[0] * deltas.shape[1], deltas.shape[2])

        portfolio = portfolio[:-1,:]
        portfolio = portfolio.reshape(portfolio.shape[0] * portfolio.shape[1], 1)

        new_input = np.concatenate([new_input,deltas,portfolio],axis=1)

    else: 

        new_latent = latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])

        #One lag for deltas
        deltas[1:,:,:] = deltas[:-1,:,:] 
        deltas[0,:,:] = 0
        deltas = deltas.reshape(deltas.shape[0] * deltas.shape[1], deltas.shape[2])
        new_latent = np.abs(deltas)

        portfolio = portfolio[:-1,:]
        portfolio = portfolio.reshape(portfolio.shape[0] * portfolio.shape[1], 1)

        new_input = np.concatenate([new_input,portfolio],axis=1)

    # Generate random indices
    random_indices = np.random.choice(new_input.shape[0], 1000000, replace=False)

    # Initialize a matrix to store the Spearman correlation coefficients
    correlation_matrix = np.zeros([new_input.shape[1], new_latent.shape[1]])

    # Compute Spearman correlation for each pair of columns
    for i in range(new_input.shape[1]):
        for j in range(new_latent.shape[1]):
            # Compute Spearman correlation between column i of array1 and column j of array2
            corr, _ = spearmanr(new_input[random_indices, i], new_latent[random_indices, j])
            correlation_matrix[i, j] = corr

    # Initialize a matrix to store the Spearman correlation coefficients
    bootstraping = 20
    independence_matrix = np.zeros([new_input.shape[1], new_latent.shape[1],bootstraping])
    for s in range(bootstraping):
        random_indices = np.random.choice(new_input.shape[0], 1000, replace=False)
        # Compute Spearman correlation for each pair of columns
        for i in range(new_input.shape[1]):
            for j in range(new_latent.shape[1]):
                # Compute Spearman correlation between column i of array1 and column j of array2
                corr = empirical_copula(new_input[random_indices, i], new_latent[random_indices, j])
                independence_matrix[i, j,s] = corr

    independence_matrix_0 = 0
    independence_matrix_1 = 0
    for i in range(bootstraping):
        independence_matrix_0 += independence_matrix[:,:,i]
    independence_matrix_0 = independence_matrix_0/bootstraping
    for i in range(bootstraping):
        independence_matrix_1 += (independence_matrix[:,:,i]-independence_matrix_0)**2
    independence_matrix_1 = np.sqrt(independence_matrix_1/(bootstraping-1))

    k = 4.5
    limit_inf = np.maximum(independence_matrix_0-k*independence_matrix_1,0)
    limit_sup = np.minimum(independence_matrix_0+k*independence_matrix_1,1)

    # Update matplotlib settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.serif'] = ['Times New Roman']  # or another serif font you prefer

    # Labels
    if autoencoder == True:
        x_labels = [r'$S_{t}$', r'$\tau$', r'$\beta_{1}$', r'$\beta_{2}$', r'$\beta_{3}$', r'$\beta_{4}$', r'$\beta_{5}$',
                    r'$h_{1}$', r'$h_{2}$', r'$h_{3}$', r'$h_{4}$', r'$h_{5}$', r'$h_{R}$', r'$P$', r'$\Delta_{P}$', r'$\gamma_{P}$',
                    'DOTM\%', 'NM\%', 'DITM\%'] + [rf'$\mathcal{{O}}_{{{i+1}}}$' for i in range(0, deltas.shape[1]-1)] +[rf'$\delta_{{{i}}}$' for i in range(1, deltas.shape[1]+1)] + [r'$V_{t}$']
        y_labels = [rf'$L_{{{i}}}$' for i in range(1, new_latent.shape[1]+1)]
    else:
        x_labels = [r'$S_{t}$', r'$\tau$', r'$\beta_{1}$', r'$\beta_{2}$', r'$\beta_{3}$', r'$\beta_{4}$', r'$\beta_{5}$',
                    r'$h_{1}$', r'$h_{2}$', r'$h_{3}$', r'$h_{4}$', r'$h_{5}$', r'$h_{R}$', r'$P$', r'$\Delta_{P}$', r'$\gamma_{P}$'] + [rf'$\mathcal{{O}}_{{{i+1}}}$' for i in range(0, deltas.shape[1]-1)] + [r'$V_{t}$']
        y_labels = [rf'$\delta_{{{i}}}$' for i in range(1, deltas.shape[1]+1)]

        #[r'$S_{t}$', r'$\tau$', r'$\beta_{1}$', r'$\beta_{2}$', r'$\beta_{3}$', r'$\beta_{4}$', r'$\beta_{5}$',
        #            r'$h_{1}$', r'$h_{2}$', r'$h_{3}$', r'$h_{4}$', r'$h_{5}$', r'$h_{R}$', r'$P$', r'$\Delta_{P}$', r'$\gamma_{P}$',
        #            'DOTM\%', 'NM\%', 'DITM\%'] + [rf'$\mathcal{{O}}_{{{i+1}}}$' for i in range(0, deltas.shape[1]-1)] + [r'$V_{t}$']

    # Custom colormap for the first heatmap
    cmap1 = LinearSegmentedColormap.from_list("custom_cmap1", ["darkred", "black", "darkgreen"])

    # Custom colormap for the second heatmap
    cmap2 = LinearSegmentedColormap.from_list("custom_cmap2", ["black", "green", "darkgreen"])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # First heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap1, fmt=".3f",
                xticklabels=y_labels,
                yticklabels=x_labels, ax=axes[0], cbar_kws={'shrink': 0.8, 'pad': 0.02}, annot_kws={"size": 14},
                vmin=-1, vmax=1)
    axes[0].set_xlabel('Latent space', fontsize=14)
    axes[0].set_ylabel('State space', fontsize=14)

    # Second heatmap
    sns.heatmap(independence_matrix_0, annot=True, cmap=cmap2, fmt=".3f",
                xticklabels=y_labels,
                yticklabels=x_labels, ax=axes[1], cbar_kws={'shrink': 0.8, 'pad': 0.02}, annot_kws={"size": 14},
                vmin=0, vmax=1)
    axes[1].set_xlabel('Latent space', fontsize=14)

    plt.subplots_adjust(wspace=0.05)  # Reduce horizontal space between plots
    plt.tight_layout()

    plt.savefig(os.path.join(main_folder, f"data/results/Training/results_extra/plots/dependency_{name}.pdf"), format="pdf", bbox_inches="tight")
    plt.show()

    np.save(os.path.join(main_folder, f"data/results/Training/results_extra/plots/limit_inf_{name}.npy"),limit_inf)
    np.save(os.path.join(main_folder, f"data/results/Training/results_extra/plots/limit_sup_{name}.npy"),limit_sup)

    return loss_metrics, limit_inf




