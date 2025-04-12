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
from src.utils import moneyness_def


class iv_option_simulation(object):
    
    def __init__(self, time_steps = 1, time_steps_hegde_option=2, number_simulations = 1, lower_bound = 1/100, isput='BS', r = 0.026623194, q = 0.01772245, issmile=False, delta = 1/252):

        # Parameters
        self.time_steps = time_steps                             #Path time-steps
        self.number_simulations = number_simulations             #Number of paths 
        self.time_steps_hegde_option = time_steps_hegde_option   #Path time-steps of the main derivative
        self.delta = delta                                       #Daily time step
        self.q = q                                               #Dividend rate as a constant
        self.r = r                                               #Interest rate as a constant
        self.lower_bound = lower_bound                           #Indicator function to cilp volatility values
        self.flag = 0                                            #Flag to determine if the maturity is equal to the time steps
        self.isput = isput                                       #Indicator function to determine vanilla european option type
        self.issmile = issmile                                   #Indicator function to determine what kind of delta we want to compute
        self.mask_smirk = None                                   #Indicator function to mask high values of beta5 coefficient of the IV surface
    
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
        
    def implied_volatility_simulation(self, S, B, K):

      """Function to simulate option paths 

          Parameters
          ----------
          S : numpy array with underlying asset price simulation
          B : numpy array with IV coefficients simulation
          K : strike option price

          Returns
          -------
          IV : simulation of implied volatility
          TT : time-to-maturity series

    """
      Stock_paths = S
      betas_simulation = B

      if (self.time_steps_hegde_option == self.time_steps):
          self.time_steps = self.time_steps - 1
          Stock_paths = Stock_paths[:,:-1]
          self.flag = 1
      elif self.flag == 1:
          Stock_paths = Stock_paths[:,:-1]
      
      #Simulation of the implied volatility
      implied_volatility_simulation_2 = np.zeros([self.number_simulations,(self.time_steps+1)])

      #Compute forward prices for all simulations
      time_to_maturity_2 = np.array(sorted((np.arange(self.time_steps_hegde_option)+1),reverse=True))[0:(self.time_steps+1)]/252
      interest_rates_difference = self.r - self.q
      forward_price_2 = Stock_paths[:,:]*np.exp(time_to_maturity_2*interest_rates_difference)

      #Compute Moneyness for all simulations
      moneyness_2 = np.log(forward_price_2/K[:, np.newaxis])*(1/np.sqrt(time_to_maturity_2))
      
      for time_step in range(self.time_steps+1):
          implied_volatility_simulation_2[:,time_step] = self.volatility( moneyness_2[:,time_step],time_to_maturity_2[time_step],betas_simulation[:,time_step,:])
      
      IV, TT = implied_volatility_simulation_2, time_to_maturity_2

      return IV, TT
    
    def Black_Scholes_price(self, S, K, TT, sigma):
        
        if (self.time_steps_hegde_option == self.time_steps):
            self.time_steps = self.time_steps - 1
            S_1 = S.copy()
            S = S[:,:-1]
            self.flag = 1
        elif self.flag == 1:
            S_1 = S.copy()
            S = S[:,:-1]
        
        d1 = (np.log(S/K[:, np.newaxis])+(self.r-self.q+(sigma**2)/2)*TT)/(sigma*np.sqrt(TT))
        d2 = d1-sigma*np.sqrt(TT)
        if self.isput==False:
            price = S*np.exp(-self.q*TT)*norm.cdf(d1)-K[:, np.newaxis]*np.exp(-self.r*TT)*norm.cdf(d2)
        else:
            price = K[:, np.newaxis]*np.exp(-self.r*TT)*norm.cdf(-d2)-S*np.exp(-self.q*TT)*norm.cdf(-d1)

        #Include the terminal payoff of the option    
        if self.flag == 1:
          if self.isput == False:
            payoff = np.maximum((S_1[:,-1]-K),0)
          else:
            payoff = np.maximum(-1*(S_1[:,-1]-K),0)
          S_1[:,:-1] = price
          S_1[:,-1]  = payoff
          price = S_1
        
        return price
    
    def deltas_f(self, S, K, TT, sigma, tc, B, T_max = 5):
        
        #Define dimension in terms of maturity and time steps for the simulation
        if (self.time_steps_hegde_option == self.time_steps):
          self.time_steps = self.time_steps - 1
          S = S[:,:-1]
          B = B[:,:-1,:]
          self.flag = 1
        elif self.flag == 1:
          S = S[:,:-1]
          B = B[:,:-1,:]
        
        #Compute deltas based on smile-implied formulas from JIVR model
        if self.issmile == "SI":
          #Compute Moneyness for all simulations
          interest_rates_difference = self.r - self.q
          forward_price = S*np.exp(TT*interest_rates_difference)
          moneyness = np.log(forward_price/K[:, np.newaxis])*(1/np.sqrt(TT))

          #Computation of preliminary quantities
          delta_1 = moneyness/sigma+(1/2)*(sigma*np.sqrt(TT))
          M_mask = (moneyness)*(moneyness<0)
          sigma_derivative = B[:,:,2]*(moneyness>=0)+ B[:,:,2]*(1-((np.exp(2*M_mask)-1)/(np.exp(2*M_mask)+1))**2)*(moneyness<0)+B[:,:,3]*2*moneyness*np.exp(-1*(moneyness**2))*np.log(TT[0]/T_max)-B[:,:,4]*81*(M_mask**2)*np.exp(27*(M_mask**3))*np.log(TT[0]/T_max)*(moneyness<0)
          
          #Deltas
          deltas = (norm.cdf(delta_1)+norm.pdf(delta_1)*sigma_derivative)*np.exp(-1*self.q*TT) 

        #Compute Black-Scholes delta or Black-Scholes Leland delta
        else:
          if self.issmile == "BSL":
            new_implied_volatility_simulation = sigma*np.sqrt(1+(tc*np.sqrt(2/(1*math.pi)))/(sigma*np.sqrt(self.delta)))
          else:
            new_implied_volatility_simulation = sigma
              
          d1 = (np.log(S[:,:]/K[:, np.newaxis])+((self.r-self.q+(new_implied_volatility_simulation**2)/2)*TT))/(new_implied_volatility_simulation*np.sqrt(TT))

          deltas = norm.cdf(d1)*np.exp(-1*self.q*TT)

        if self.isput==False:
            deltas = deltas
        else:
            deltas = deltas - np.exp(-1*self.q*TT)
            
        return deltas
    
    def gammas_f(self, S, K, TT, sigma, tc, B, T_max = 5):
        
        #Define dimension in terms of maturity and time steps for the simulation
        if (self.time_steps_hegde_option == self.time_steps):
          self.time_steps = self.time_steps - 1
          S = S[:,:-1]
          B = B[:,:-1,:]
          self.flag = 1
        elif self.flag == 1:
          S = S[:,:-1]
          B = B[:,:-1,:]

        #Compute gammas based on smile-implied formulas from JIVR model
        if self.issmile == "SI":

          #Mask for the Smirk (beta 5)
          if self.mask_smirk is None:
              mask_beta_5 = 1
          else:
              mask_beta_5 = (B[:,:,4]<=0)
          B[:,:,4] = B[:,:,4]*mask_beta_5

          #Compute Moneyness for all simulations
          interest_rates_difference = self.r - self.q
          forward_price = S*np.exp(TT*interest_rates_difference)
          moneyness = np.log(forward_price/K[:, np.newaxis])*(1/np.sqrt(TT))

          #Computation of preliminary quantities
          delta_1 = moneyness/sigma+(1/2)*(sigma*np.sqrt(TT))
          M_mask = (moneyness)*(moneyness<0)
          sigma_derivative = B[:,:,2]*(moneyness>=0)+ B[:,:,2]*(1-((np.exp(2*M_mask)-1)/(np.exp(2*M_mask)+1))**2)*(moneyness<0)+B[:,:,3]*2*moneyness*np.exp(-1*(moneyness**2))*np.log(TT[0]/T_max)-B[:,:,4]*81*(M_mask**2)*np.exp(27*(M_mask**3))*np.log(TT[0]/T_max)*(moneyness<0)
          sigma_second_derivative = -1*B[:,:,2]*8*np.exp(2*M_mask)*((np.exp(2*M_mask)-1)/((np.exp(2*M_mask)+1)**3))*(moneyness<0)+B[:,:,3]*2*(1-moneyness**2)*np.exp(-1*(moneyness**2))*np.log(TT[0]/T_max)-B[:,:,4]*(162-6561*(M_mask**3))*M_mask*np.exp(27*(M_mask**3))*np.log(TT[0]/T_max)*(moneyness<0)
          delta_1_derivative = (1/sigma)-((moneyness/(sigma**2))-(1/2*np.sqrt(TT)))*sigma_derivative

          S_0 = S[:,0]
          gammas = norm.pdf(delta_1)*(delta_1_derivative-delta_1*delta_1_derivative*sigma_derivative+sigma_second_derivative)*(np.exp(-1*self.q*TT)/(S_0[:, np.newaxis]*np.sqrt(TT)))
        
        #Compute Black-Scholes delta or Black-Scholes Leland delta
        else:

          if self.issmile == "BSL":
            new_implied_volatility_simulation = sigma*np.sqrt(1+(tc*np.sqrt(2/math.pi))/(sigma*np.sqrt(self.delta)))
          else:
            new_implied_volatility_simulation = sigma
              
          d1 = (np.log(S[:,:]/K[:, np.newaxis])+((self.r-self.q+(new_implied_volatility_simulation**2)/2)*TT))/(new_implied_volatility_simulation*np.sqrt(TT))
              
          gammas = (norm.pdf(d1) / (S[:,:]*new_implied_volatility_simulation*np.sqrt(TT)))*np.exp(-1*self.q*np.sqrt(TT))
        
        return gammas
    
    def vega(self, S, K, TT, sigma, tc, B, T_max = 5):
       
      #Define dimension in terms of maturity and time steps for the simulation
      if (self.time_steps_hegde_option == self.time_steps):
        self.time_steps = self.time_steps - 1
        S = S[:,:-1]
        B = B[:,:-1,:]
        self.flag = 1
      elif self.flag == 1:
        S = S[:,:-1]
        B = B[:,:-1,:]
      
      new_implied_volatility_simulation = sigma
              
      d1 = (np.log(S[:,:]/K[:, np.newaxis])+((self.r-self.q+(new_implied_volatility_simulation**2)/2)*TT))/(new_implied_volatility_simulation*np.sqrt(TT))

      vegas = S[:,:]*np.exp(-1*self.q*np.sqrt(TT))*np.sqrt(TT)*norm.pdf(d1)
       
      return vegas
    
    def vanna(self, S, K, TT, sigma, tc, B, T_max = 5):
       
      #Define dimension in terms of maturity and time steps for the simulation
      if (self.time_steps_hegde_option == self.time_steps):
        self.time_steps = self.time_steps - 1
        S = S[:,:-1]
        B = B[:,:-1,:]
        self.flag = 1
      elif self.flag == 1:
        S = S[:,:-1]
        B = B[:,:-1,:]
      
      new_implied_volatility_simulation = sigma

      d1 = (np.log(S[:,:]/K[:, np.newaxis])+((self.r-self.q+(new_implied_volatility_simulation**2)/2)*TT))/(new_implied_volatility_simulation*np.sqrt(TT))
      d2 = d1-new_implied_volatility_simulation*np.sqrt(TT)

      vannas = (-1)*np.exp(-1*self.q*np.sqrt(TT))*norm.pdf(d1)*d2/new_implied_volatility_simulation

      return vannas
    
    def vomma(self, S, K, TT, sigma, tc, B, T_max = 5):
       
      #Define dimension in terms of maturity and time steps for the simulation
      if (self.time_steps_hegde_option == self.time_steps):
        self.time_steps = self.time_steps - 1
        S = S[:,:-1]
        B = B[:,:-1,:]
        self.flag = 1
      elif self.flag == 1:
        S = S[:,:-1]
        B = B[:,:-1,:]
      
      new_implied_volatility_simulation = sigma
              
      d1 = (np.log(S[:,:]/K[:, np.newaxis])+((self.r-self.q+(new_implied_volatility_simulation**2)/2)*TT))/(new_implied_volatility_simulation*np.sqrt(TT))
      d2 = d1-new_implied_volatility_simulation*np.sqrt(TT)

      vommas = S[:,:]*np.exp(-1*self.q*np.sqrt(TT))*np.sqrt(TT)*norm.pdf(d1)*(d1*d2/new_implied_volatility_simulation)

      return vommas


def option_simulator(time_steps, time_steps_hegde_option, number_simulations, lower_bound, isput, r, q, delta, S, B, K, tc, issmile):

  """Function to simulate option paths 

        Parameters
        ----------
        time_steps               : time steps consider in the option path simulation 
        time_steps_hegde_option  : maturity of the option in days
        number_simulations       : number of simulation
        lower_bound              : indicator function to cilp volatility values
        isput                    : boolean variable to determine european option type
        r                        : anualized risk-free
        q                        : anualized dividend yield
        delta                    : step size in years
        S                        : numpy array with underlying asset price simulation
        B                        : numpy array with IV coefficients simulation
        K                        : numpy array with strike price
        tc                       : transaction cost level (0,1)
        issmile                  : Indicator function to determine what kind of delta we want to compute

        Returns
        -------
        option_price             : Option price simulation
        deltas                   : Deltas of the option simulation
        gammas                   : Gammas of the option simulation

    """
  ivf = iv_option_simulation(time_steps, time_steps_hegde_option, number_simulations, lower_bound, isput, r, q, issmile, delta)
  IV, TT = ivf.implied_volatility_simulation(S, B, K)
  option_price = ivf.Black_Scholes_price(S, K, TT, IV)
  deltas = ivf.deltas_f(S,K,TT, IV,tc, B)
  gammas = ivf.gammas_f(S,K,TT, IV,tc, B)

  return option_price, deltas, gammas, TT, IV

def hedging_intrument_simulation(temporality, time_steps_hegde_option, number_simulations, lower_bound, isput, r, q, delta, S, B, tc, issmile, moneyness, dynamics):
   
  """Function to simulate hedging instrument history 

    Parameters
    ----------
    time_steps               : time steps consider in the option path simulation 
    time_steps_hegde_option  : maturity of the option in days
    number_simulations       : number of simulation
    lower_bound              : indicator function to cilp volatility values
    isput                    : boolean variable to determine european option type
    r                        : anualized risk-free
    q                        : anualized dividend yield
    delta                    : step size in years
    S                        : numpy array with underlying asset price simulation
    B                        : numpy array with IV coefficients simulation
    K                        : numpy array with strike price
    tc                       : transaction cost level (0,1)
    issmile                  : Indicator function to determine what kind of delta we want to compute
    moneyness                : Moneyness of the hedging instrument regardless the dynamics ATM, OTM, ITM
    dynamics                 : "static" for a single static hedging instrument and "dynamic" for a new "moneyess" option each day

    Returns
    -------
    option_price             : Option price history
    deltas                   : Deltas of the option history
    gammas                   : Gammas of the option history

  """
  #Define arrays for hedging option intruments
  if dynamics == "static":
    option_price = np.zeros([temporality+1,number_simulations]) 
    deltas = np.zeros([temporality+1,number_simulations])
    gammas = np.zeros([temporality+1,number_simulations])
    TT = np.zeros([temporality+1])
    IV = np.zeros([temporality+1,number_simulations])
  elif dynamics == "dynamic":
    option_price = np.zeros([temporality+1,number_simulations,temporality]) #[time,num_simulations,option]
    deltas = np.zeros([temporality+1,number_simulations,temporality])
    gammas = np.zeros([temporality+1,number_simulations,temporality])
    TT = np.zeros([temporality+1,temporality])
    IV = np.zeros([temporality+1,number_simulations,temporality])
  elif dynamics == "semistatic":
    option_price = np.zeros([temporality+1,number_simulations,2]) #[time,num_simulations,option]
    deltas = np.zeros([temporality+1,number_simulations])
    gammas = np.zeros([temporality+1,number_simulations])
    TT = np.zeros([temporality+1,number_simulations])
    IV = np.zeros([temporality+1,number_simulations])
     
  #Define time steps based on temporality
  time_steps = temporality if temporality<=time_steps_hegde_option else time_steps_hegde_option

  #Compute a single hedging instrument for the whole hedging process (for instance only a ATM option for the whole experiment)
  if dynamics == "static":
    K = moneyness_def(S[:,0], moneyness, isput)
    option_price_h, deltas_h, gammas_h, TT_h, IV_h = option_simulator(time_steps, time_steps_hegde_option, number_simulations, lower_bound, isput, r, q, delta, S[:,:time_steps+1], B[:,:time_steps+1,:], K, tc, issmile)
    option_price[:option_price_h.shape[1],:] += np.transpose(option_price_h)
    deltas[:deltas_h.shape[1],:] += np.transpose(deltas_h)
    gammas[:gammas_h.shape[1],:] += np.transpose(gammas_h)
    TT[:TT_h.shape[0]] += TT_h
    IV[:IV_h.shape[1],:] += np.transpose(IV_h)

  #Compute a new hedging intrument every day based on the moneyness condition (for example a new ATM option every day)
  elif dynamics == "dynamic":
    for initial_day in range(temporality):
       if temporality<=time_steps_hegde_option:  
        time_steps_h = time_steps - initial_day
        limit_up = time_steps
       else:
        time_steps_h = min(time_steps,temporality-initial_day)
        limit_up = initial_day + time_steps_h
       
       K = moneyness_def(S[:,initial_day], moneyness, isput)
       option_price_h, deltas_h, gammas_h, TT_h, IV_h = option_simulator(time_steps_h, time_steps_hegde_option, number_simulations, lower_bound, isput, r, q, delta, S[:,initial_day:limit_up+1], B[:,initial_day:limit_up+1,:], K, tc, issmile)
       option_price[initial_day:limit_up+1,:,initial_day] = np.transpose(option_price_h)
       #print(time_steps_h, time_steps_hegde_option,deltas[initial_day:limit_up,:,initial_day].shape,deltas_h.shape)
       # Create a new limit up when time_steps_h==time_steps_hegde_option due to in this case deltas_h include one less value
       new_limit = limit_up if time_steps_h==time_steps_hegde_option else limit_up+1
       deltas[initial_day:new_limit,:,initial_day] = np.transpose(deltas_h)
       gammas[initial_day:new_limit,:,initial_day] = np.transpose(gammas_h)
       TT[initial_day:new_limit,initial_day] = TT_h
       IV[initial_day:new_limit,:,initial_day] = np.transpose(IV_h)

  elif dynamics == "semistatic":
     
    for i in range(number_simulations):
      flag = 1
      counter = 0
      time = None
      while flag ==1 :
          #At this point the while starts
          time = time_steps if counter == 0 else time
          limit_up = time if counter == 0 else limit_up
          initial_day = 0 if counter == 0 else limit_up
          time_steps_h = time if counter == 0 else time_steps-initial_day

          #Moneyness computation
          #Compute moneyness of linked to each option
          S_k = S[i,initial_day:time_steps+1]
          K = moneyness_def(S_k[0], moneyness, isput)
          TT_k  = np.array(sorted((np.arange(time_steps_hegde_option)+1),reverse=True))[0:(time_steps_h+1)]/252
          interest_rates_difference = r - q
          forward_price = S_k*np.exp(TT_k*interest_rates_difference)
          moneyness_k = np.log(forward_price/K)*(1/np.sqrt(TT_k))

          #Compute time where we are out of moneyness interval
          time = np.where(((moneyness_k>0.2) + (moneyness_k<-0.2))==True)[0]

          if (len(time) != 0):
            if time[0] == time_steps:
                time = []

          if len(time) == 0:
              limit_up = time_steps
              option_price_h, deltas_h, gammas_h, TT_h, IV_h = option_simulator(time_steps_h, time_steps_hegde_option, 1, lower_bound, isput, r, q, delta, S[i,initial_day:limit_up+1][np.newaxis,:], B[i,initial_day:limit_up+1,:][np.newaxis,:,:], K[np.newaxis], tc, issmile)
              option_price[initial_day:limit_up+1,i,0] = option_price_h[0,:]
              deltas[initial_day:limit_up+1,i] = deltas_h[0,:]
              gammas[initial_day:limit_up+1,i] = gammas_h[0,:]
              TT[initial_day:limit_up+1,i] = TT_h
              IV[initial_day:limit_up+1,i] = IV_h[0,:]
              flag = 0
          else:
              time = time[0]
              limit_up = time if counter == 0 else limit_up + time
              time_steps_h = time

              option_price_h, deltas_h, gammas_h, TT_h, IV_h = option_simulator(time_steps_h, time_steps_hegde_option, 1, lower_bound, isput, r, q, delta, S[i,initial_day:limit_up+1][np.newaxis,:], B[i,initial_day:limit_up+1,:][np.newaxis,:,:], K[np.newaxis], tc, issmile)
              option_price[initial_day:limit_up+1,i,0] = option_price_h[0,:]
              option_price[limit_up,i,1] = option_price[limit_up,i,0]
              deltas[initial_day:limit_up+1,i] = deltas_h[0,:]
              gammas[initial_day:limit_up+1,i] = gammas_h[0,:]
              TT[initial_day:limit_up+1,i] = TT_h
              IV[initial_day:limit_up+1,i] = IV_h[0,:]

              counter += 1
     
  return option_price, deltas, gammas, TT, IV

def hedging_instruments(temporality, hedging_intruments_maturity, option_types, option_moneyness, number_simulations, lower_bound, r, q, delta, S, B, tc, issmile, dynamics):
   
   """Function to simulate hedging instruments 

    Parameters
    ----------
    time_steps               : time steps consider in the option path simulation 
    time_steps_hegde_option  : maturity of the option in days
    number_simulations       : number of simulation
    lower_bound              : indicator function to cilp volatility values
    isput                    : boolean variable to determine european option type
    r                        : anualized risk-free
    q                        : anualized dividend yield
    delta                    : step size in years
    S                        : numpy array with underlying asset price simulation
    B                        : numpy array with IV coefficients simulation
    K                        : numpy array with strike price
    tc                       : transaction cost level (0,1)
    issmile                  : Indicator function to determine what kind of delta we want to compute
    moneyness                : Moneyness of the hedging instrument regardless the dynamics ATM, OTM, ITM
    dynamics                 : "static" for a single static hedging instrument and "dynamic" for a new "moneyess" option each day

    Returns
    -------
    option_price             : Option price history
    deltas                   : Deltas of the option history
    gammas                   : Gammas of the option history

  """
   #Define dimension of hedging instrments
   hedging_instrument_dim = len(hedging_intruments_maturity)
   
   #Define arrays for hedging option intruments
   if dynamics == "static":
    option_price = np.zeros([temporality+1,number_simulations,hedging_instrument_dim]) 
    deltas = np.zeros([temporality+1,number_simulations,hedging_instrument_dim])
    gammas = np.zeros([temporality+1,number_simulations,hedging_instrument_dim])
    TT = np.zeros([temporality+1,hedging_instrument_dim])
    IV = np.zeros([temporality+1,number_simulations,hedging_instrument_dim])
   elif dynamics == "dynamic":
    option_price = np.zeros([temporality+1,number_simulations,temporality,hedging_instrument_dim]) #[time ,num_simulations, option_of_day, hedging_instrument]
    deltas = np.zeros([temporality+1,number_simulations,temporality,hedging_instrument_dim])
    gammas = np.zeros([temporality+1,number_simulations,temporality,hedging_instrument_dim])
    TT = np.zeros([temporality+1,temporality,hedging_instrument_dim])
    IV = np.zeros([temporality+1,number_simulations,temporality,hedging_instrument_dim])
   elif dynamics == "semistatic":
    option_price = np.zeros([temporality+1,number_simulations,2,hedging_instrument_dim]) 
    deltas = np.zeros([temporality+1,number_simulations,hedging_instrument_dim])
    gammas = np.zeros([temporality+1,number_simulations,hedging_instrument_dim])
    TT = np.zeros([temporality+1,number_simulations,hedging_instrument_dim])
    IV = np.zeros([temporality+1,number_simulations,hedging_instrument_dim])

   #Fill up the hedging instruments history
   for i in range(hedging_instrument_dim):

     option_price_h, deltas_h, gammas_h, TT_h, IV_h = hedging_intrument_simulation(temporality, hedging_intruments_maturity[i], number_simulations, lower_bound, option_types[i], r, q, delta, S, B, tc[1], issmile, option_moneyness[i], dynamics)
     if dynamics == "static":
      option_price[:,:,i] = option_price_h
      deltas[:,:,i] = deltas_h
      gammas[:,:,i] = gammas_h
      TT[:,i] = TT_h
      IV[:,:,i] = IV_h
     elif dynamics == "dynamic":
      option_price[:,:,:,i] = option_price_h
      deltas[:,:,:,i] = deltas_h
      gammas[:,:,:,i] = gammas_h
      TT[:,:,i] = TT_h
      IV[:,:,:,i] = IV_h
     elif dynamics == "semistatic":
      option_price[:,:,:,i] = option_price_h
      deltas[:,:,i] = deltas_h
      gammas[:,:,i] = gammas_h
      TT[:,:,i] = TT_h
      IV[:,:,i] = IV_h

   #Option price of each new instrument available in the market
   if dynamics == "static":
      new_option_price = option_price[:-1,:,:]
   elif dynamics == "dynamic":
      new_option_price = np.zeros([temporality,number_simulations,len(hedging_intruments_maturity)])
      for o in range(len(hedging_intruments_maturity)):
        for i in range(temporality):
            new_option_price[i,:,o] = option_price[i,:,i,o]
   elif dynamics == "semistatic":
      new_option_price = option_price[:-1,:,0,:]
     
   return option_price, deltas, gammas, TT, IV, new_option_price


def hedged_portfolio(temporality, number_simulations, hedged_options_maturities, option_types, moneyness, positions, lower_bound, r, q, delta, S, B, tc, issmile):
   
   """Function to simulate the hedged portfolio

    Parameters
    ----------
    temporality                : time steps consider in the experiment
    number_simulations         : number of simulation 
    hedged_options_maturities  : list - maturity of the options in the portfolio
    option_type                : list - option types
    moneyness                  : list - moneyness of the options in the portfolio
    positions                  : list - number of shares of each option
    lower_bound                : indicator function to cilp volatility values
    r                          : anualized risk-free
    q                          : anualized dividend yield
    delta                      : step size in years
    S                          : numpy array with underlying asset price simulation
    B                          : numpy array with IV coefficients simulation
    tc                         : transaction cost level (0,1)
    issmile                    : Indicator function to determine what kind of delta we want to compute

    Returns
    -------
    portfolio_array            : Portfolio history (cash-flows, portfolio value, deltas, gammas)
    moneyness_array            : moneyness statistics according to each option (moneyness for single hedged intrument, proportions for portfolios)
    

  """
   
   #Define portfolio numpy array
   portfolio_array = np.zeros([number_simulations,temporality+1,4]) #Five corresponds to the hedger cash flows (payments and risk primes), portfolio value, deltas and gammas
   
   #Identify number of actions
   number_of_actions =  len(hedged_options_maturities)

   #Define moneyness array
   dimension = 3 if number_of_actions>1 else 1
   moneyness_array = np.zeros([number_simulations,temporality+1,dimension])

   #Condition for hedging
   if max(hedged_options_maturities)<temporality:
      print("Error - at least one of the hedged instruments has to have maturity greater or equal than temporality")
   else:
    #Simulate options paths
     for i in range(number_of_actions):
        K = moneyness_def(S[:,0], moneyness[i], option_types[i])
        time_steps = temporality if temporality<=hedged_options_maturities[i] else hedged_options_maturities[i]
        option_price_h, deltas_h, gammas_h, TT_h, _ = option_simulator(time_steps, hedged_options_maturities[i], number_simulations, lower_bound, option_types[i], r, q, delta, S[:,:time_steps+1], B[:,:time_steps+1,:], K, tc[0], issmile)

        #Compute moneyness of linked to each option
        S_k = S[:,:time_steps] if (hedged_options_maturities[i] == time_steps) else S
        B_k = B[:,:time_steps,:] if (hedged_options_maturities[i] == time_steps) else B
        TT_k = TT_h
        interest_rates_difference = r - q
        forward_price = S_k*np.exp(TT_k*interest_rates_difference)
        moneyness_k = np.log(forward_price/K[:, np.newaxis])*(1/np.sqrt(TT_k))
        if dimension==1:
          moneyness_array[:,:moneyness_k.shape[1],0]=moneyness_k
        else:
          if option_types[i] == False:
              DOTM = (moneyness_k < -0.2)
              NM = (-0.2<= moneyness_k) & (moneyness_k <= 0.2)
              DITM = (0.2 < moneyness_k)
          else:
              DOTM = (0.2 < moneyness_k)
              NM = (-0.2<= moneyness_k) & (moneyness_k <= 0.2)
              DITM = (moneyness_k < -0.2)
          moneyness_array[:,:moneyness_k.shape[1],0] += DOTM*positions[i]
          moneyness_array[:,:moneyness_k.shape[1],1] += NM*positions[i]
          moneyness_array[:,:moneyness_k.shape[1],2] += DITM*positions[i]

        #Compute payments for positions closed in the next if
        if temporality>hedged_options_maturities[i]:
          portfolio_array[:,option_price_h.shape[1]-1,0] += -1*positions[i]*option_price_h[:,-1]

        #Compute portafolio history (features for deep hedging and delta-gamma hedging)
        portfolio_array[:,:option_price_h.shape[1],1] += positions[i]*option_price_h
        portfolio_array[:,:deltas_h.shape[1],2] += positions[i]*deltas_h
        portfolio_array[:,:gammas_h.shape[1],3] += positions[i]*gammas_h

     #Last update of the moneyness array (compute proportions when it applies)
     if dimension==1:
      moneyness_array = moneyness_array 
     else:
      total_cases = moneyness_array[:,:,0]+moneyness_array[:,:,1]+moneyness_array[:,:,2]
      for i in range(dimension):
        if max(hedged_options_maturities)<=temporality:
          moneyness_array[:,:-1,i] = moneyness_array[:,:-1,i]/total_cases[:,:-1]
        else:
          moneyness_array[:,:,i] = moneyness_array[:,:,i]/total_cases[:,:]

     #Portfolio value at the end of the temporality (close hedged portfolio)
     portfolio_array[:,-1,0] += -1*portfolio_array[:,-1,1]

     #Risk prime received at the begining of the hedging period
     portfolio_array[:,0,0] += portfolio_array[:,0,1]
    
   return portfolio_array, moneyness_array


#######################################
### Functions for greeks evaluation ###
#######################################

def option_simulator_2(time_steps, time_steps_hegde_option, number_simulations, lower_bound, isput, r, q, delta, S, B, K, tc, issmile):

  """Function to simulate option paths 

        Parameters
        ----------
        time_steps               : time steps consider in the option path simulation 
        time_steps_hegde_option  : maturity of the option in days
        number_simulations       : number of simulation
        lower_bound              : indicator function to cilp volatility values
        isput                    : boolean variable to determine european option type
        r                        : anualized risk-free
        q                        : anualized dividend yield
        delta                    : step size in years
        S                        : numpy array with underlying asset price simulation
        B                        : numpy array with IV coefficients simulation
        K                        : numpy array with strike price
        tc                       : transaction cost level (0,1)
        issmile                  : Indicator function to determine what kind of delta we want to compute

        Returns
        -------
        option_price             : Option price simulation
        deltas                   : Deltas of the option simulation
        gammas                   : Gammas of the option simulation

    """
  ivf = iv_option_simulation(time_steps, time_steps_hegde_option, number_simulations, lower_bound, isput, r, q, issmile, delta)
  IV, TT = ivf.implied_volatility_simulation(S, B, K)
  option_price = ivf.Black_Scholes_price(S, K, TT, IV)
  deltas = ivf.deltas_f(S,K,TT, IV,tc, B)
  gammas = ivf.gammas_f(S,K,TT, IV,tc, B)
  vegas  = ivf.vega(S,K,TT, IV,tc, B)
  vannas = ivf.vanna(S,K,TT, IV,tc, B)
  vommas = ivf.vomma(S,K,TT, IV,tc, B)

  return option_price, deltas, gammas, vegas, vannas, vommas, TT, IV


def hedged_portfolio_greeks(temporality, number_simulations, hedged_options_maturities, option_types, moneyness, positions, lower_bound, r, q, delta, S, B, tc, issmile):
   
   """Function to simulate the hedged portfolio

    Parameters
    ----------
    temporality                : time steps consider in the experiment
    number_simulations         : number of simulation 
    hedged_options_maturities  : list - maturity of the options in the portfolio
    option_type                : list - option types
    moneyness                  : list - moneyness of the options in the portfolio
    positions                  : list - number of shares of each option
    lower_bound                : indicator function to cilp volatility values
    r                          : anualized risk-free
    q                          : anualized dividend yield
    delta                      : step size in years
    S                          : numpy array with underlying asset price simulation
    B                          : numpy array with IV coefficients simulation
    tc                         : transaction cost level (0,1)
    issmile                    : Indicator function to determine what kind of delta we want to compute

    Returns
    -------
    portfolio_array            : Portfolio history (cash-flows, portfolio value, deltas, gammas)
    moneyness_array            : moneyness statistics according to each option (moneyness for single hedged intrument, proportions for portfolios)
    

  """
   
   #Define portfolio numpy array
   portfolio_array = np.zeros([number_simulations,temporality+1,7]) #Five corresponds to the hedger cash flows (payments and risk primes), portfolio value, deltas and gammas
   
   #Identify number of actions
   number_of_actions =  len(hedged_options_maturities)

   #Define moneyness array
   dimension = 3 if number_of_actions>1 else 1
   moneyness_array = np.zeros([number_simulations,temporality+1,dimension])

   #Condition for hedging
   if max(hedged_options_maturities)<temporality:
      print("Error - at least one of the hedged instruments has to have maturity greater or equal than temporality")
   else:
    #Simulate options paths
     for i in range(number_of_actions):
        K = moneyness_def(S[:,0], moneyness[i], option_types[i])
        time_steps = temporality if temporality<=hedged_options_maturities[i] else hedged_options_maturities[i]
        option_price_h, deltas_h, gammas_h, vegas_h, vannas_h, vommas_h, TT_h, _ = option_simulator_2(time_steps, hedged_options_maturities[i], number_simulations, lower_bound, option_types[i], r, q, delta, S[:,:time_steps+1], B[:,:time_steps+1,:], K, tc[0], issmile)

        #Compute moneyness of linked to each option
        S_k = S[:,:time_steps] if (hedged_options_maturities[i] == time_steps) else S
        B_k = B[:,:time_steps,:] if (hedged_options_maturities[i] == time_steps) else B
        TT_k = TT_h
        interest_rates_difference = r - q
        forward_price = S_k*np.exp(TT_k*interest_rates_difference)
        moneyness_k = np.log(forward_price/K[:, np.newaxis])*(1/np.sqrt(TT_k))
        if dimension==1:
          moneyness_array[:,:moneyness_k.shape[1],0]=moneyness_k
        else:
          if option_types[i] == False:
              DOTM = (moneyness_k < -0.2)
              NM = (-0.2<= moneyness_k) & (moneyness_k <= 0.2)
              DITM = (0.2 < moneyness_k)
          else:
              DOTM = (0.2 < moneyness_k)
              NM = (-0.2<= moneyness_k) & (moneyness_k <= 0.2)
              DITM = (moneyness_k < -0.2)
          moneyness_array[:,:moneyness_k.shape[1],0] += DOTM*positions[i]
          moneyness_array[:,:moneyness_k.shape[1],1] += NM*positions[i]
          moneyness_array[:,:moneyness_k.shape[1],2] += DITM*positions[i]

        #Compute payments for positions closed in the next if
        if temporality>hedged_options_maturities[i]:
          portfolio_array[:,option_price_h.shape[1]-1,0] += -1*positions[i]*option_price_h[:,-1]

        #Compute portafolio history (features for deep hedging and delta-gamma hedging)
        portfolio_array[:,:option_price_h.shape[1],1] += positions[i]*option_price_h
        portfolio_array[:,:deltas_h.shape[1],2] += positions[i]*deltas_h
        portfolio_array[:,:gammas_h.shape[1],3] += positions[i]*gammas_h
        portfolio_array[:,:deltas_h.shape[1],4] += positions[i]*vegas_h
        portfolio_array[:,:gammas_h.shape[1],5] += positions[i]*vannas_h
        portfolio_array[:,:deltas_h.shape[1],6] += positions[i]*vommas_h

     #Last update of the moneyness array (compute proportions when it applies)
     if dimension==1:
      moneyness_array = moneyness_array 
     else:
      total_cases = moneyness_array[:,:,0]+moneyness_array[:,:,1]+moneyness_array[:,:,2]
      for i in range(dimension):
        if max(hedged_options_maturities)<=temporality:
          moneyness_array[:,:-1,i] = moneyness_array[:,:-1,i]/total_cases[:,:-1]
        else:
          moneyness_array[:,:,i] = moneyness_array[:,:,i]/total_cases[:,:]

     #Portfolio value at the end of the temporality (close hedged portfolio)
     portfolio_array[:,-1,0] += -1*portfolio_array[:,-1,1]

     #Risk prime received at the begining of the hedging period
     portfolio_array[:,0,0] += portfolio_array[:,0,1]
    
   return portfolio_array, moneyness_array

















