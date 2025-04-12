"""
Usage:
    1. cd src
    2. python3 features/jivr_simulation.py 
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
from random import sample
import random

class implied_volatily_surface_vec(object):
    
    """Class to the implied volatility surface model, recalibrate it, and use it to predict future dynamics
    
    Parameters
    ----------
    config_file : simulation settings for the JIVR model and the underlying asset and 
    
    """
    
    def __init__(self,config_file):
        # Parameters
        self.config_file = config_file
        self.time_steps = config_file['simulation']['number_days']                #Path time-steps
        self.number_simulations = config_file['simulation']['number_simulations_batch'] #Number of simulations per batch 
        self.delta = eval(config_file['simulation']['size_step'])                 #Daily time step
        self.q = config_file['simulation']['q']                                   #Dividend rate as a constant
        self.r = config_file['simulation']['r']                                   #Interest rate as a constant
        self.lower_bound = config_file['simulation']['vol_lower_bound']           #Indicator function to cilp volatility values 
        self.S_0 = config_file['simulation']['stock_price']                       #Initial value of the stock price
        self.simulations = config_file['simulation']['number_simulations']        #Number of simulated paths
        self.seed = config_file['simulation']['seed']                             #Seed to ensure replicability 
    
    def cumulative(self, parameters, z):
        
        """Function that compute the cumulant generating function of NIG variables 

        Parameters
        ----------
        parameters : data frame - parameters of the econometric model
        z : data frame - real value

        Returns
        -------
        psy : cumulant value at point z

        """
        
        phi = parameters.iloc[13,5]
        zeta = parameters.iloc[12,5] 
        
        psy = (phi**2/(phi**2+zeta**2))*((-1)*zeta*z+phi**2-phi*np.sqrt(phi**2+zeta**2-((zeta+z)**2)))
        
        return psy
        
    def data_preparation(self, parameters,betas_series,h_series,Y_series):
    
        """Function that provide multidimensional vector for updating the parameters 

        Parameters
        ----------
        parameters : data frame - parameters of the econometric model
        betas_series : data frame - historical data of the beta parameters
        h_series : data frame historical data of the volvol of the econometric model
        Y_series : data frame - historical data of the returns

        Returns
        -------
        beta_t : beta parameters at time t
        beta_t_minus : beta parameters at time t-1
        h_t : simulated volatility at time t
        e_t : NIG values at time t

        """
        betas_t = np.array(betas_series.iloc[betas_series.shape[0]-1,:])
        betas_t_minus_1 = np.array(betas_series.iloc[betas_series.shape[0]-2,:])
        betas_t_minus_2 = np.array(betas_series.iloc[betas_series.shape[0]-3,:])
        h_t = np.array(h_series.iloc[h_series.shape[0]-1,0:5])
        h_t_Y = np.array(h_series.iloc[h_series.shape[0]-1,[5]])
        Y_t = Y_series.iloc[Y_series.shape[0]-1,0]
        
        """"Compute NIG variables for the last observation for betas"""
        e_t = []
        for i in range(0,5):
            e_t.append((betas_t[i]-parameters.iloc[0,i]-np.sum(parameters.iloc[1:6,i]*betas_t_minus_1)-parameters.iloc[6,1]*betas_t_minus_2[1]*(i==1))/np.sqrt(h_t[i]*self.delta))
        e_t = np.array(e_t).reshape(-1)
        
        
        """"Compute NIG variables for the return"""
        z = np.sqrt(h_t_Y*self.delta)
        e_t_Y = (Y_t-self.cumulative(parameters, (-1)*parameters.iloc[0,5]*z)+self.cumulative(parameters, (1-parameters.iloc[0,5])*z))/z
        
        return betas_t, betas_t_minus_1, h_t, e_t, h_t_Y, e_t_Y
    
        
    def volatility(self,M,tau,beta,T_conv = 0.25,T_max = 5):
    
        """Function that provides the volatility surface model

        Parameters
        ----------
        M : sigle value - Moneyness
        tau : single value - Time to maturity
        beta : numpy.ndarray - parameters 
        T_conv : location of a fast convexity change in the IV term structure with respect to time to maturity
        T_max : single value - Maximal maturity represented by the model

        Returns
        -------
        Volatility 

        """
        
        Long_term_ATM_Level = beta[:,0]
        Time_to_maturity_slope = beta[:,1]*np.exp(-1*np.sqrt(tau/T_conv))
        M_plus = (M<0)*M
        Moneyness_slope = (beta[:,2]*M)*(M>=0)+ (beta[:,2]*((np.exp(2*M_plus)-1)/(np.exp(2*M_plus)+1)))*(M<0)
        Smile_attenuation = beta[:,3]*(1-np.exp(-1*(M**2)))*np.log(tau/T_max)
        Smirk = (beta[:,4]*(1-np.exp((3*M_plus)**3))*np.log(tau/T_max))*(M<0)
        
        if self.lower_bound is None:
            volatility = Long_term_ATM_Level + Time_to_maturity_slope + Moneyness_slope + Smile_attenuation + Smirk
        else:
            volatility = np.maximum(Long_term_ATM_Level + Time_to_maturity_slope + Moneyness_slope + Smile_attenuation + Smirk,self.lower_bound)

        return volatility
        
    def betas_returns_simulations(self, parameters, betas_series, h_series, Y_series, NIG):
        
        """Function that provide multidimensional vector for updating the parameters 

        Parameters
        ----------
        parameters : data frame - parameters of the econometric model
        betas_series : data frame - historical data of the beta parameters
        h_series : data frame historical data of the volvol of the econometric model
        Y_series : data frame - historical data of the returns

        Returns
        -------
        
        betas_simulation : numpy.ndarray of dimension s,m,n
        returns_simulation : numpy.ndarray of dimension s,m

        """
        
        betas_t, betas_t_minus_1, h_t, e_t, h_t_Y, e_t_Y = self.data_preparation(parameters,betas_series,h_series,Y_series)

        #Definition of arrays to store data
        e_t_plus = np.zeros([self.number_simulations,self.time_steps+1,6])
        betas_plus = np.zeros([self.number_simulations,self.time_steps+2,5])
        h_t_plus = np.zeros([self.number_simulations,self.time_steps+1,6])
        returns = np.zeros([self.number_simulations,self.time_steps])

        #Arrays inicialization
        e_t_plus[:,0,:] = np.repeat(np.append(e_t,e_t_Y).reshape(1,6), self.number_simulations,axis=0)
        h_t_plus[:,0,:] = np.repeat(np.append(h_t,h_t_Y).reshape(1,6), self.number_simulations,axis=0)
        betas_plus[:,0:2,:]=np.repeat(np.concatenate((betas_t_minus_1,betas_t), axis=0).reshape(1,2,5), self.number_simulations,axis=0)
        
        #NIG variables
        if NIG.shape[0] >= self.number_simulations*self.time_steps:
            NIG = NIG[:self.number_simulations*self.time_steps,:].reshape([self.number_simulations,self.time_steps,6])
            e_t_plus[:,1:,:] = NIG
        else:
            print("The NIG_simulation file does not contain enough simulations to simulate these paths")
            print("The NIG_simulations contains " + str(int(NIG.shape[0])) + " simulations of " + str(int(self.number_simulations*self.time_steps)))
            sys.exit()
            
        #Simulate Betas and returns
        for t in range(self.time_steps):
            #Return simulation
            V_t_Y = (parameters.iloc[8,5]*self.volatility(0,1/12,betas_plus[:,t+1,:]))**2
            h_t_plus[:,t+1,5] = V_t_Y + parameters.iloc[9,5]*(h_t_plus[:,t,5]-V_t_Y) + parameters.iloc[10,5]*h_t_plus[:,t,5]*(e_t_plus[:,t,5]**2-1-2*e_t_plus[:,t,5]*parameters.iloc[11,5])
            returns[:,t] = (self.r-self.q)*self.delta + self.cumulative(parameters, (-1)*parameters.iloc[0,5]*np.sqrt(h_t_plus[:,t+1,5]*self.delta)) - self.cumulative(parameters, (1-parameters.iloc[0,5])*np.sqrt(h_t_plus[:,t+1,5]*self.delta)) + np.sqrt(h_t_plus[:,t+1,5]*self.delta)*e_t_plus[:,t+1,5]
      
            #Beta 1 simulation
            U_t = (parameters.iloc[8,0]*self.volatility(0,1/12,betas_plus[:,t+1,:]))**2
            h_t_plus[:,t+1,0] = U_t + parameters.iloc[9,0]*(h_t_plus[:,t,0]-U_t) + parameters.iloc[10,0]*h_t_plus[:,t,0]*(e_t_plus[:,t,0]**2-1-2*e_t_plus[:,t,0]*parameters.iloc[11,0])
            betas_plus[:,t+2,0] = (parameters.iloc[0,0]+np.matmul(betas_plus[:,t+1,:],np.array(parameters.iloc[1:6,0]).reshape(5,1)).reshape(-1)+np.sqrt(h_t_plus[:,t+1,0]*self.delta)*e_t_plus[:,t+1,0])

            #Rest of the betas simulation
            for i in range(1,5):
                sigma = parameters.iloc[7,i]/np.sqrt(252)
                h_t_plus[:,t+1,i] = sigma**2 + parameters.iloc[9,i]*(h_t_plus[:,t,i]-sigma**2) + parameters.iloc[10,i]*h_t_plus[:,t,i]*(e_t_plus[:,t,i]**2-1-2*e_t_plus[:,t,i]*parameters.iloc[11,i])
                betas_plus[:,t+2,i] = parameters.iloc[0,i]+np.matmul(betas_plus[:,t+1,:],np.array(parameters.iloc[1:6,i]).reshape(5,1)).reshape(-1)+parameters.iloc[6,1]*betas_plus[:,t,1]*(i==1)+np.sqrt(h_t_plus[:,t+1,i]*self.delta)*e_t_plus[:,t+1,i]
        
        betas_simulation = betas_plus[:,1:,:]
        returns_simulation = returns
        
        return betas_simulation, returns_simulation, h_t_plus
    
    def jivr_simulation(self):
        
        random.seed(self.seed)

        #Raw data and NIG simulation
        parameters = pd.read_csv(os.path.join(main_folder,'data','raw','Parameters_2.csv'), index_col=0)
        betas_series = pd.read_csv(os.path.join(main_folder,'data','raw','Time_series_of_betas_random.csv'), index_col=0)
        h_series = pd.read_csv(os.path.join(main_folder,'data','raw','Time_series_of_h_random.csv'), index_col=0)
        Y_series = pd.read_csv(os.path.join(main_folder,'data','raw','SP500_random.csv'), index_col=0)
        NIG = np.load(os.path.join(main_folder,'data','interim','NIG_simulation.npy'))

        #JVIR model simulation - Features simulation
        betas_random   = np.zeros([6289*self.number_simulations,self.time_steps+1,5])
        h_t_random     = np.zeros([6289*self.number_simulations,self.time_steps+1,6])
        returns_random = np.zeros([6289*self.number_simulations,self.time_steps])

        #Random index for NIG simulations
        index    = list(np.linspace(0,NIG.shape[0]-1,NIG.shape[0]).astype('int'))
        print("-- JIVR simulation starts --")
        print("-- Progress of JIVR simulation: ", end='', flush=True)
        for i in range(6289): #6289
            sample_2 = sample(index,self.number_simulations*self.time_steps)
            NIG_2 = NIG[sample_2,:]
            
            betas_simulation, returns_simulation, h_t_plus = self.betas_returns_simulations(parameters, betas_series.iloc[:6291-i,:], h_series.iloc[:6291-i,:], Y_series.iloc[:6291-i,:],NIG_2)
            
            betas_random[i*(self.number_simulations):(i+1)*(self.number_simulations),:,:] = betas_simulation
            h_t_random[i*(self.number_simulations):(i+1)*(self.number_simulations),:,:]   = h_t_plus
            returns_random[i*(self.number_simulations):(i+1)*(self.number_simulations),:] = returns_simulation
            
            print("\r--Progress of JIVR simulation: {:.2%}".format((i+1)/6289), end='', flush=True)
        print("\n--Simulation of JIVR features completed--")
        #Simulate Stock Paths
        print("-- Simulation of stock price --")
        Stock_paths = self.S_0*np.cumprod(np.exp(returns_random),axis=1)
        Stock_paths = np.insert(Stock_paths,0,self.S_0,axis=1)

        #Clean simulation to avoid NaN values 
        big  = np.where((Stock_paths>400).sum(axis=1)>0)[0]
        nans = np.where(np.isnan(Stock_paths).sum(axis=1)>0)[0]
        sorted_sample = np.arange(1006240).astype('int')
        sorted_sample = np.setdiff1d(np.setdiff1d(sorted_sample,big.astype('int')),nans.astype('int')).astype('int')
        random_sample = sample(list(sorted_sample),self.simulations)

        pd.DataFrame(random_sample).to_csv(os.path.join(main_folder,'data','processed','training',f"random_sample_{self.time_steps}.csv"))

        np.save(os.path.join(main_folder,'data','processed','training',f"Stock_paths__random_f_{self.time_steps}.npy"),Stock_paths[random_sample,:])
        np.save(os.path.join(main_folder,'data','processed','training',f"Betas_simulation__random_f_{self.time_steps}.npy"),betas_random[random_sample,:,:])
        np.save(os.path.join(main_folder,'data','processed','training',f"H_simulation__random_f_{self.time_steps}.npy"),h_t_random[random_sample,:,:]) 
        np.save(os.path.join(main_folder,'data','processed','training',f"Returns_random__random_f_{self.time_steps}.npy"),returns_random[random_sample,:])

        print("-- Simulation completed - JIVR features stored in ../data/processed/--")

        return 
    
if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file = load_config(os.path.join(main_folder,'cfgs','config_simulation.yml'))
    jivr_model = implied_volatily_surface_vec(config_file)
    jivr_model.jivr_simulation()