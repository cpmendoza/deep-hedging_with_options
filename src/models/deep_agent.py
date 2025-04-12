"""
Usage:
    1. cd src
    2. python 
"""
#

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow.compat.v1 as tf
import random
import datetime as dt
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

class StrictlyPositive(tf.keras.constraints.Constraint):
    def __init__(self, min_value=1e-6):
        self.min_value = min_value

    def __call__(self, w):
        # Ensure all elements are strictly positive by clamping to a small positive value
        return tf.maximum(w, self.min_value)

    def get_config(self):
        return {'min_value': self.min_value}

class DeepAgent(object):
    """
    Inputs:
    network        : neural network architechture {LSTM,RNN-FNN,FFNN}
    nbs_point_traj : if [S_0,...,S_N] ---> nbs_point_traj = N+1
    batch_size     : size of mini-batch
    nbs_input      : number of features (without considerint V_t)
    nbs_units      : number of neurons per layer
    nbs_assets     : dimension of the output layer (number of hedging instruments)
    lambda_m       : regularization parameter for soft-constraint in lagrange multiplier
    constraint_max : Lower bound of the output layer activation function
    loss_type      : loss function for the optimization procedure {CVaR,SMSE,MSE}
    lr             : learning rate hyperparameter of the Adam optimizer
    dropout_par:   : dropout regularization parameter [0,1]
    isput          : condition to determine the option type for the hedging error {True,False}
    prepro_stock   : {Log, Log-moneyness, Nothing} - what transformation was used for stock prices
    name           : name to store the trained model
    process        : determine training or inference procedure

    # Disclore     : Class adapted from https://github.com/alexandrecarbonneau/Deep-Equal-Risk-Pricing-of-Financial-Derivatives-with-Multiple-Hedging-Instruments/blob/main/Example%20of%20ERP%20with%20deep%20hedging%20multi%20hedge%20-%20Final.ipynb
    """
    def __init__(self, network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lambda_m, constraint_min, loss_type, lr, dropout_par, autoencoder, nbs_latent_spc, lambda_auto, name, process):
        tf.compat.v1.disable_eager_execution()
        ops.reset_default_graph()

        # 0) Deep hedging parameters parameters
        self.network        = network
        self.nbs_point_traj = nbs_point_traj
        self.batch_size     = batch_size
        self.nbs_input      = nbs_input
        self.nbs_units      = nbs_units
        self.nbs_assets     = nbs_assets
        self.loss_type      = loss_type
        self.lr             = lr
        self.dropout_par    = dropout_par
        self.deltas         = tf.zeros(shape = [nbs_point_traj-1, batch_size, nbs_assets], dtype=tf.float32)  # array to store position in the hedging instruments
        
        # 1) Soft-constraint parameters
        self.lambda_m       = lambda_m
        self.constraint_min = constraint_min
        self.condition      = tf.zeros(shape = [batch_size], dtype=tf.float32)

        # 2) Autoencoder parameters
        self.autoencoder    = autoencoder
        self.nbs_latent_spc = nbs_latent_spc
        self.lambda_auto    = lambda_auto
        self.loss_auto      = tf.zeros(shape = [1], dtype=tf.float32)
        self.loss_auto_vec  = tf.zeros(shape = [nbs_point_traj,batch_size])
        self.latent         = tf.zeros(shape = [nbs_point_traj, batch_size, self.nbs_latent_spc])
        self.portfolio      = tf.zeros(shape = [nbs_point_traj, batch_size])
        #autoencoder, nbs_latent_spc, lambda_auto

        # 3) Placeholder for deep hedging elements        
        self.input                = tf.placeholder(tf.float32, [nbs_point_traj, batch_size, nbs_input])         # normalized prices and features
        self.underlying           = tf.placeholder(tf.float32, [nbs_point_traj, batch_size])                    # underlying asset prices
        self.cashflow             = tf.placeholder(tf.float32, [nbs_point_traj, batch_size])                    # cash flows of hedged portfolio
        self.riskaversion         = tf.placeholder(tf.float32)                                                  # CVaR confidence level (alpha in (0,1))
        self.risk_free_factor     = tf.placeholder(tf.float32)                                                  # risk-free rate factor exp(delta*r)
        self.dividendyield_factor = tf.placeholder(tf.float32)                                                  # dividend rate factor exp(delta*d)
        self.transaction_cost     = tf.placeholder(tf.float32, [self.nbs_assets])                               # transaction cost rate (tc in (0,1))
        self.rebalancing_const    = tf.placeholder(tf.float32, [self.nbs_assets])                               # threshold for rebalancing
        self.hedging_instruments  = tf.placeholder(tf.float32, [nbs_point_traj, batch_size, 2, self.nbs_assets-1]) # hedging instruments prices

        non_neg_constraint = StrictlyPositive(min_value=0)#tf.keras.constraints.NonNeg() #1e-5
        self.rebalancing_const_1   = tf.compat.v1.get_variable(name='rebalancing_const', shape=[1], initializer=tf.compat.v1.constant_initializer([0.001]), trainable = True, constraint=non_neg_constraint)  # Add the constraint here

        # 4) Autoencoder
        if (self.autoencoder == True):
            layer_a_1 = tf.layers.Dense(36, tf.nn.relu)
            layer_a_2 = tf.layers.Dense(18, tf.nn.relu)
            layer_a_3 = tf.layers.Dense(9, tf.nn.relu)
            layer_a_4 = tf.layers.Dense(self.nbs_latent_spc, None)
            layer_a_5 = tf.layers.Dense(9, tf.nn.relu)
            layer_a_6 = tf.layers.Dense(18, tf.nn.relu)
            layer_a_7 = tf.layers.Dense(36, tf.nn.relu)
            layer_a_8 = tf.layers.Dense(self.nbs_input+1+self.nbs_assets, tf.nn.relu) # +1 portfolio value and + self.nbs_assets for deltas

        # 5) Network architechture for the deep hedging algorithm
        if (self.network == "LSTM"):
          # 5.1.1) Four LSTM cells (the dimension of the hidden state and the output is the same)
          layer_1 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units) 
          layer_2 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          layer_3 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          layer_4 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          # 5.1.2) Output layer of dimension nbs_assets (outputs the position in the hedging instruments)
          layer_out = tf.layers.Dense(self.nbs_assets, None)
        elif (self.network == "RNNFNN"):
          # 5.2.1) Two LSTM cells (the dimension of the hidden state and the output is the same)
          #      Two regular layers with RELU activation function and dropout regularization per layer
          layer_1 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units) 
          layer_2 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          layer_3 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_3 = tf.keras.layers.Dropout(self.dropout_par)
          layer_4 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_4 = tf.keras.layers.Dropout(self.dropout_par)
          # 5.2.2) Output layer of dimension nbs_assets (outputs the position in the hedging instruments)
          layer_out = tf.layers.Dense(self.nbs_assets, None)
        elif (self.network == "FFNN"):
          # 5.3.1) Four regular layers with RELU activation function and dropout regularization per layer
          layer_1 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_1 = tf.keras.layers.Dropout(self.dropout_par)
          layer_2 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_2 = tf.keras.layers.Dropout(self.dropout_par)
          layer_3 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_3 = tf.keras.layers.Dropout(self.dropout_par)
          layer_4 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_4 = tf.keras.layers.Dropout(self.dropout_par)
          # 5.3.2) Output layer of dimension one (outputs the position in the underlying)
          layer_out = tf.layers.Dense(self.nbs_assets, None)

        # 6) Compute hedging strategies for all time-steps
          
        # First cash flow and first possible action 
        V_t = self.cashflow[0,:]
        self.portfolio = tf.expand_dims(V_t,axis=0)                         
        self.layer_prev = tf.zeros(shape = [batch_size,self.nbs_assets], dtype=tf.float32)
        condition = self.input[0,:,13]

        # Based on initial information, we start rebalancing hedging portfolio
        for t in range(self.nbs_point_traj-1):
            
            input_t = tf.concat([self.input[t,:,:], tf.expand_dims(V_t, axis = 1)], axis=1)
            input_t = tf.concat([input_t, self.layer_prev], axis=1)
            layer_l = input_t

            if (self.autoencoder == True):
                #Autoencoder
                layer = layer_a_1(input_t)
                layer = layer_a_2(layer)
                layer = layer_a_3(layer)
                layer_l = layer_a_4(layer)
                layer = layer_a_5(layer_l)
                layer = layer_a_6(layer)
                layer = layer_a_7(layer)
                layer = layer_a_8(layer)
                #loss of autoencoder
                if(t==0):
                    self.latent = tf.expand_dims(layer_l,axis=0)
                    self.loss_auto_vec = tf.expand_dims(tf.math.reduce_sum(tf.square(input_t-layer),axis=1),axis=0)
                else: 
                    self.latent = tf.concat([self.latent, tf.expand_dims(layer_l,axis=0)], axis = 0)
                    self.loss_auto_vec = tf.concat([self.loss_auto_vec, tf.expand_dims(tf.math.reduce_sum(tf.square(input_t-layer),axis=1),axis=0)], axis = 0)

            input_t = layer_l if self.network == "FFNN" else tf.expand_dims(layer_l , axis = 1)
            
            #RL Agent
            if (self.network == "LSTM"):
                # forward prop at time 't'
                layer = layer_1(input_t)
                layer = layer_2(tf.expand_dims(layer, axis = 1))
                layer = layer_3(tf.expand_dims(layer, axis = 1))
                layer = layer_4(tf.expand_dims(layer, axis = 1))
                layer = layer_out(layer)

            elif (self.network == "RNNFNN"):
                # forward prop at time 't'
                layer = layer_1(input_t)
                layer = layer_2(tf.expand_dims(layer, axis = 1))
                layer = layer_3(layer)
                layer = layer_drop_3(layer)
                layer = layer_4(layer)
                layer = layer_drop_4(layer)
                layer = layer_out(layer)

            else:
                # forward prop at time 't'
                layer = layer_1(input_t)
                layer = layer_drop_1(layer)
                layer = layer_2(layer)
                layer = layer_drop_2(layer)
                layer = layer_3(layer)
                layer = layer_drop_3(layer)
                layer = layer_4(layer)
                layer = layer_drop_4(layer)
                layer = layer_out(layer)   

            
            # Desition for rebalancing - no-trade region
            sharpness=self.rebalancing_const[0]
            epsilon=self.rebalancing_const[1]
            x = tf.concat([tf.expand_dims(self.underlying[t,:]*tf.constant(0,tf.float32), axis = 1),self.hedging_instruments[t,:,1,:]],axis=1)
            y = tf.constant(0, tf.float32)
            indicator_1 = tf.sigmoid(sharpness * ((x - y) - epsilon))
            x = tf.math.abs(layer-self.layer_prev)
            x = tf.math.reduce_sum(x,axis=1)
            y = self.rebalancing_const_1
            indicator_2 = tf.sigmoid(sharpness * (x - y))

            # Smooth approximation of logical OR by taking the maximum
            smooth_indicator = tf.maximum(indicator_1, indicator_2[:, tf.newaxis])
            
            # Comment this condition if transaction cost is 0
            if process == 'inference':
                smooth_indicator = tf.sigmoid(sharpness * (smooth_indicator - 0.5))

            # Apply smooth interpolation to ensure differentiability
            layer = smooth_indicator * layer + (1 - smooth_indicator) * self.layer_prev

            # Compile trading strategies
            if (t==0):
                # At t = 0, need to expand the dimension to have [nbs_point_traj, batch_size, nbs_assets]
                self.deltas = tf.expand_dims(layer,axis=0)                      # [1, batch_size, nbs_assets]
                self.cost   = tf.zeros(shape = [batch_size],dtype=tf.float32)   # Vector to store the hedging strategy transaction cost
                #Compute transaction cost of all hedging instruments
                self.cost += tf.math.abs(self.deltas[t,:,0]*self.underlying[t,:])*self.transaction_cost[0]
                for a in range(self.nbs_assets-1):
                    self.cost += tf.math.abs(self.deltas[t,:,a+1]*self.hedging_instruments[t,:,0,a])*self.transaction_cost[1]
            else:

                #Store the rest of the hedging positions
                self.deltas = tf.concat([self.deltas, tf.expand_dims(layer, axis = 0)], axis = 0)
                self.cost   = tf.zeros(shape = [batch_size], dtype=tf.float32)
                #Compute transaction cost of all hedging instruments
                self.cost +=  tf.math.abs(self.underlying[t,:]*(self.deltas[t,:,0]-self.deltas[t-1,:,0]))*self.transaction_cost[0]
                for a in range(self.nbs_assets-1):
                    self.cost += tf.where(tf.math.greater(self.hedging_instruments[t,:,1,a],tf.constant(0,tf.float32)),
                                   (tf.math.abs(self.hedging_instruments[t,:,1,a]*(self.deltas[t-1,:,a+1]))+tf.math.abs(self.hedging_instruments[t,:,0,a]*(self.deltas[t,:,a+1])))*self.transaction_cost[1],
                                    tf.math.abs(self.hedging_instruments[t,:,0,a]*(self.deltas[t,:,a+1]-self.deltas[t-1,:,a+1]))*self.transaction_cost[1])

            # Compute the portoflio value for the next period
            V_t_pre = V_t

            #Previous position
            self.aux_0 = tf.zeros(shape = [batch_size], dtype=tf.float32)
            self.aux_0 += self.deltas[t,:,0]*self.underlying[t,:]
            for a in range(self.nbs_assets-1):
                 self.aux_0 += self.deltas[t,:,a+1]*self.hedging_instruments[t,:,0,a]
            phi_0   = V_t_pre - self.aux_0 - self.cost

            #New portfolio value
            self.aux = tf.zeros(shape = [batch_size], dtype=tf.float32)
            for a in range(self.nbs_assets):
                if a==0:
                  self.aux += self.deltas[t,:,a]*self.underlying[t+1,:]*self.dividendyield_factor
                else:
                  self.aux += tf.where(tf.math.greater(self.hedging_instruments[t+1,:,1,a-1],tf.constant(0,tf.float32)),
                                   self.deltas[t,:,a]*self.hedging_instruments[t+1,:,1,a-1],
                                    self.deltas[t,:,a]*self.hedging_instruments[t+1,:,0,a-1])

            V_t = phi_0*self.risk_free_factor + self.aux + self.cashflow[t+1,:]

            #Compute soft tracking error constraint
            if (t+1)<(self.nbs_point_traj-1):
                self.condition += tf.cast(tf.math.greater(self.input[t+1,:,13]-V_t, condition*self.constraint_min), tf.float32)

            self.layer_prev   = tf.zeros(shape = [batch_size,self.nbs_assets], dtype=tf.float32)
            self.layer_prev  += layer
            self.portfolio    = tf.concat([self.portfolio, tf.expand_dims(V_t,axis=0)], axis = 0)

        # 6) Compute hedging errors for each path
        
        # Hedging cost for liquidate the position in hedging options
        self.cost   = tf.zeros(shape = [batch_size],dtype=tf.float32)   # Vector to store the hedging strategy transaction cost
        #Compute transaction cost of all hedging instruments
        for a in range(self.nbs_assets-1):
            self.cost += tf.math.abs(self.deltas[t,:,a+1]*self.hedging_instruments[t+1,:,0,a])*self.transaction_cost[1]
        # Substract final cost to close postions in options
        V_t = V_t - self.cost

        # Compute hedging error
        self.hedging_err = 0 - V_t

        # 6.1) Compute soft-constraint (tracking error constraint)
        self.condition = tf.cast(tf.math.greater(self.condition,tf.constant(0,tf.float32)), dtype=tf.float32)
        self.condition_mean = tf.reduce_mean(self.condition)

        # 7) Compute the loss function on the batch of hedging error
        # - This is the empirical cost functions estimated with a mini-batch
        if (self.loss_type == "CVaR"):
            self.loss = tf.reduce_mean(tf.sort(self.hedging_err)[tf.cast(self.riskaversion*self.batch_size,tf.int32):])
        elif (self.loss_type == "MSE"):
            self.loss = tf.reduce_mean(tf.square(self.hedging_err)) 
        elif (self.loss_type == "SMSE"):
            self.loss = tf.reduce_mean(tf.square(tf.nn.relu(self.hedging_err))) 

        # Autoencoder inclusion
        self.loss_auto_vec = tf.reshape(self.loss_auto_vec, [-1])
        self.loss_auto = tf.reduce_mean(self.loss_auto_vec)
        self.loss = self.loss + self.lambda_auto*self.loss_auto + self.lambda_m*self.condition_mean
       
        # 8) SGD step with the adam optimizer
        optimizer  = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train = optimizer.minimize(self.loss)

        # 9) Save the model
        self.saver      = tf.train.Saver()
        self.model_name = name   # name of the neural network to save

    # Function to compute the CVaR_{alpha} outside the optimization, i.e. at the end of each epoch in this case
    def loss_out_optim(self, hedging_err, alpha, loss_type):
        if (loss_type == "CVaR"):
            loss = np.mean(np.sort(hedging_err)[int(alpha*hedging_err.shape[0]):])
        elif (loss_type == "MSE"):
            loss = np.mean(np.square(hedging_err))
        elif (loss_type == "SMSE"):
            loss = np.mean(np.square(np.where(hedging_err>0,hedging_err,0)))
        return loss

    # ---------------------------------------------------------------------------------------#
    # Function to call the deep hedging algorithm batch-wise
    """
    Input:
    train_input          : Training set (normalized stock price and features)
    underlying_train     : Underlying asset prices for training set
    underlying_test      : Underlying asset prices for validation set
    HO_train             : Prices of hedging instruments in the training set
    HO_test              : Prices of hedging instruments in the validation set
    cash_flows_train     : Cash flows of hedged portfolio for training set
    cash_flows_test      : Cash flows of hedged portfolio for validation set
    risk_free_factor     : Risk-free rate update factor exp(h*r)
    dividendyield_factor : Dividend yield update factor exp(h*d)
    transaction_cost     : Proportional transaction cost [0,5/10000,5/1000,1/100]
    riskaversion         : CVaR confidence level (0,1)
    test_input           : Test set (normalized stock price and features) 
    sess                 : tensorflow session
    epochs               : Number of epochs, training iterations
    """
    def train_deephedging(self, train_input, underlying_train, underlying_test, HO_train, HO_test, cash_flows_train, cash_flows_test, risk_free_factor, dividendyield_factor, transaction_cost, rebalancing_const, riskaversion, test_input, sess, epochs):
                               #  paths,         prices_train,     prices_test,    V_0,    V_test,             strikes,                     disc_batch,    dividend_batch,     transacion_cost, riskaversion, paths_valid, sess, epochs
        sample_size       = train_input.shape[1]               # total number of paths in the train set
        sample_size_valid = test_input.shape[1]
        batch_size        = self.batch_size
        idx               = np.arange(sample_size)       # [0,1,...,sample_size-1]
        idx_valid         = np.arange(sample_size_valid)
        start             = dt.datetime.now()            # Time-to-train
        self.loss_epochs  = 9999999*np.ones((epochs,6))      # Store the loss at the end of each epoch for the train
        valid_loss_best   = 999999999
        epoch             = 0

        # Loop for each epoch until the maximum number of epochs
        while (epoch < epochs):
            hedging_err_train = []  # Store hedging errors obtained for one complete epoch
            hedging_err_valid = []
            autoencoder_train = []  # Store errors of autoencoder
            autoencoder_valid = []
            constraint_train = []  # Store soft-constraint vector
            constraint_valid = []
            
            np.random.shuffle(idx)  # Randomize the dataset (not useful in this case since dataset is simulated iid)

            # loop over each batch size
            for i in range(int(sample_size/batch_size)):

                # Indexes of paths used for the mini-batch
                indices = idx[i*batch_size : (i+1)*batch_size]

                # SGD step
                _, hedging_err, loss_auto, loss_condition, contstraint = sess.run([self.train, self.hedging_err,
                                                                       self.loss_auto_vec, self.condition, self.rebalancing_const_1],
                                               {self.input                 : train_input[:,indices,:],
                                                self.underlying            : underlying_train[:,indices],
                                                self.cashflow              : cash_flows_train[:,indices],
                                                self.riskaversion          : riskaversion,
                                                self.risk_free_factor      : risk_free_factor,
                                                self.dividendyield_factor  : dividendyield_factor,
                                                self.transaction_cost      : transaction_cost,
                                                self.rebalancing_const     : rebalancing_const,
                                                self.hedging_instruments   : HO_train[:,indices,:]})

                hedging_err_train.append(hedging_err)
                autoencoder_train.append(loss_auto)
                constraint_train.append(loss_condition)

            # 2) Evaluate performance on the valid set - we don't train
            for i in range(int(sample_size_valid/batch_size)):
                indices_valid = idx_valid[i*batch_size : (i+1)*batch_size]
                hedging_err_v, loss_auto_t, loss_condition_t, contstraint = sess.run([self.hedging_err,
                                                                         self.loss_auto_vec, self.condition, self.rebalancing_const_1],
                                               {self.input                 : test_input[:,indices_valid,:],
                                                self.underlying            : underlying_test[:,indices_valid],
                                                self.cashflow              : cash_flows_test[:,indices_valid],
                                                self.riskaversion          : riskaversion,
                                                self.risk_free_factor      : risk_free_factor,
                                                self.dividendyield_factor  : dividendyield_factor,
                                                self.transaction_cost      : transaction_cost,
                                                self.rebalancing_const     : rebalancing_const,
                                                self.hedging_instruments   : HO_test[:,indices_valid,:]})

                hedging_err_valid.append(hedging_err_v)
                autoencoder_valid.append(loss_auto_t)
                constraint_valid.append(loss_condition_t)

            # 3) Store the loss on the train and valid sets after each epoch
            self.loss_epochs[epoch,0] = self.loss_out_optim(np.concatenate(hedging_err_train), riskaversion, self.loss_type)
            self.loss_epochs[epoch,1] = self.loss_out_optim(np.concatenate(hedging_err_valid), riskaversion, self.loss_type) #, axis=1
            self.loss_epochs[epoch,2] = np.mean(np.concatenate(autoencoder_train))
            self.loss_epochs[epoch,3] = np.mean(np.concatenate(autoencoder_valid))
            self.loss_epochs[epoch,4] = np.mean(np.concatenate(constraint_train))
            self.loss_epochs[epoch,5] = np.mean(np.concatenate(constraint_valid))

            model_saved = 0
            # 4) Test if best epoch so far on valid set; if so, save model parameters.
            if((self.loss_epochs[epoch,1] < valid_loss_best)): #& (self.loss_epochs[epoch,1]>0)
                valid_loss_best = self.loss_epochs[epoch,1]
                self.saver.save(sess, self.model_name + '.ckpt')
                model_saved = 1
                
            # Print the CVaR value at the end of each epoch
            if (epoch+1) % 1 == 0:
                if (model_saved ==1):
                    print('Epoch %d, Time elapsed:'% (epoch+1), dt.datetime.now()-start, " ---- Model saved")
                else:
                    print('Epoch %d, Time elapsed:'% (epoch+1), dt.datetime.now()-start)
                print('  Train - %s: %.3f, Auto: %.3f, Soft_cons: %.3f' % (self.loss_type,
                                                            self.loss_epochs[epoch,0],self.loss_epochs[epoch,2],self.loss_epochs[epoch,4]))
                print('  Valid - %s: %.3f, Auto: %.3f, Soft_cons: %.3f' % (self.loss_type,
                                                            self.loss_epochs[epoch,1],self.loss_epochs[epoch,3],self.loss_epochs[epoch,5]))
                contstraint = np.array(contstraint)
                formatted = ", ".join([f"Condition {i+1}: {x:.10f}" for i, x in enumerate(contstraint)])
                print(f"  {formatted}")

            epoch+=1  # increment the epoch

        # End of training
        print("---Finished training results---")
        print('Time elapsed:', dt.datetime.now()-start)

        # Return the learning curve
        return self.loss_epochs


    # Function which will call the deep hedging optimization batchwise
    def training(self, train_input, underlying_train, underlying_test, HO_train, HO_test, cash_flows_train, cash_flows_test, risk_free_factor, dividendyield_factor, transaction_cost, rebalancing_const, riskaversion, test_input, sess, epochs):
        sess.run(tf.global_variables_initializer())
        loss_train_epoch = self.train_deephedging(train_input, underlying_train, underlying_test, HO_train, HO_test, cash_flows_train, cash_flows_test, risk_free_factor, dividendyield_factor, transaction_cost, rebalancing_const, riskaversion, test_input, sess, epochs)
        return loss_train_epoch

    # ---------------------------------------------------------------------- #
    # Function to compute the hedging strategies of a trained neural network
    # - Doesn't train the neural network, only outputs the hedging strategies
    def predict(self, train, underlying, HO, cash_flows, risk_free_factor, dividendyield_factor, transaction_cost, rebalancing_const, riskaversion, sess):
        sample_size = train.shape[1]
        batch_size  = self.batch_size
        idx         = np.arange(sample_size)  # [0,1,...,sample_size-1]
        start       = dt.datetime.now()     # compute time
        strategy_pred = [] # hedging strategies
        latent_pred = []
        portfolio_pred = []

        # loop over sample size to do one complete epoch
        for i in range(int(sample_size/batch_size)):

            # mini-batch of paths (even if not training to not get memory issue)
            indices = idx[i*batch_size : (i+1)*batch_size]
            portfolio, strategy, latent, constraint = sess.run([self.portfolio, self.deltas, self.latent, self.rebalancing_const_1],
                                    {self.input                 : train[:,indices,:],
                                     self.underlying            : underlying[:,indices],
                                     self.cashflow              : cash_flows[:,indices],
                                     self.riskaversion          : riskaversion,
                                     self.risk_free_factor      : risk_free_factor,
                                     self.dividendyield_factor  : dividendyield_factor,
                                     self.transaction_cost      : transaction_cost,
                                     self.rebalancing_const     : rebalancing_const,
                                     self.hedging_instruments   : HO[:,indices,:]})

            # Append the batch of hedging strategies
            strategy_pred.append(strategy)
            latent_pred.append(latent)
            portfolio_pred.append(portfolio)

        return np.concatenate(strategy_pred,axis=1), np.concatenate(latent_pred,axis=1),  np.concatenate(portfolio_pred,axis=1), np.array(constraint)

    def restore(self, sess, checkpoint):
        self.saver.restore(sess, checkpoint)


def train_network(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lambda_m, constraint_min,
                   loss_type, lr, dropout_par, train_input, underlying_train, underlying_test, HO_train, HO_test,
                     cash_flows_train, cash_flows_test, risk_free_factor,dividendyield_factor, transaction_cost,
                       riskaversion, test_input, epochs, display_plot, id, autoencoder, nbs_latent_spc, lambda_auto, rebalancing_const):
    
    # Function to train deep hedging algorithm
    """
    Input:
    network              : Neural network architecture {"LSTM","RNNFNN","FFNN"} 
    nbs_point_traj       : time steps 
    batch_size           : batch size {296,1000}
    nbs_input            : number of features
    nbs_units            : neurons per layer/cell
    nbs_assets           : number of hedging intruments
    constraint_min       : minimum value of the portfolio for soft constraint
    lambda_m             : labmda for lagrange multiplier (soft constraint)
    autoencoder          : autoencoder component included
    nbs_latent_spc       : dimension of the latent space
    lambda_auto          : regularization parameter to ponderate autoencoder loss function
    loss_type            : loss function {"CVaR","MSE","SMSE"}
    lr                   : learning rate of the Adam optimizer
    dropout_par          : dropout regularization parameter 
    transaction_cost     : Proportional transaction cost [0,5/10000,5/1000,1/100]
    riskaversion         : CVaR confidence level (0,1)
    epochs               : Number of epochs, training iterations
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
    display_plot         : Display plot of training and validation loss 

    Output:
    loss_train_epoch : Loss history per epochs

    """

    owd = os.getcwd()
    try:
        os.chdir(os.path.join(main_folder, f"models/Random_{nbs_point_traj-1}/TC_{transaction_cost[0]*100}"))

        auto_name = f"auto{str(nbs_latent_spc)}_{str(lambda_auto)}" if autoencoder==True else "noauto"
        transaction_cost_name = f"TC_{str('_'.join(map(str, [x * 100 for x in transaction_cost])))}" if sum(transaction_cost)!= 0 else str(0)
        rebalance_name = f"learned_reebalance_integral"

        if loss_type == "CVaR":
            name = f"Deep_agent_{network}_{auto_name}_dropout_{str(int(dropout_par*100))}_{loss_type}_{str(int(riskaversion*100))}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{lambda_m}"
        else:
            name = f"Deep_agent_{network}_{auto_name}_dropout_{str(int(dropout_par*100))}_{loss_type}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{lambda_m}"

        
        # Compile the neural network
        process = 'training'
        rl_network = DeepAgent(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lambda_m, constraint_min, loss_type, lr, dropout_par, autoencoder, nbs_latent_spc, lambda_auto, name, process)
                             
        print("-------------------------------------------------------------")
        print(name)
        print("-------------------------------------------------------------")

        # Start training
        start = dt.datetime.now()
        print("-----------------------Training start------------------------")
        with tf.Session() as sess:
            loss_train_epoch = rl_network.training(train_input, underlying_train, underlying_test, HO_train, HO_test, cash_flows_train, cash_flows_test, risk_free_factor, dividendyield_factor, transaction_cost, rebalancing_const, riskaversion, test_input, sess, epochs)
        print('---Training end---')

        if display_plot == True:
            # Plot the learning curve on the train set, i.e. the CVaR on the train set at the end of each epoch
            lin_nb_epoch = np.linspace(1, loss_train_epoch.shape[0], loss_train_epoch.shape[0])
            fig = plt.figure(figsize=(5, 5), dpi=100)
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,0],label="Train error")
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,1],label="Test error")
            plt.title(f"Loss function: {loss_type}")
            plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            plt.show()
    finally:
        #change dir back to original working directory (owd)
        os.chdir(owd)
    #np.save(os.path.join(main_folder, f"data/results/Training/results_extra/loss_functions/loss_{name}.npy"),loss_train_epoch)

    return loss_train_epoch

def retrain_network(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lambda_m, constraint_min,
                   loss_type, lr, dropout_par, train_input, underlying_train, underlying_test, HO_train, HO_test,
                     cash_flows_train, cash_flows_test, risk_free_factor,dividendyield_factor, transaction_cost,
                       riskaversion, test_input, epochs, display_plot, id, autoencoder, nbs_latent_spc, lambda_auto, rebalancing_const):

    # Function to retrain deep hedging algorithm
    """
    Input:
    network              : Neural network architecture {"LSTM","RNNFNN","FFNN"}
    nbs_point_traj       : time steps 
    batch_size           : batch size {296,1000}
    nbs_input            : number of features
    nbs_units            : neurons per layer/cell
    nbs_assets           : number of hedging intruments
    constraint_min       : minimum value of the portfolio for soft constraint
    lambda_m             : labmda for lagrange multiplier (soft constraint)
    autoencoder          : autoencoder component included
    nbs_latent_spc       : dimension of the latent space
    lambda_auto          : regularization parameter to ponderate autoencoder loss function
    loss_type            : loss function {"CVaR","MSE","SMSE"}
    lr                   : learning rate of the Adam optimizer
    dropout_par          : dropout regularization parameter 
    transaction_cost     : Proportional transaction cost [0,5/10000,5/1000,1/100]
    riskaversion         : CVaR confidence level (0,1)
    epochs               : Number of epochs, training iterations
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
    display_plot         : Display plot of training and validation loss 

    Output:
    loss_train_epoch : Loss history per epochs

    """


    owd = os.getcwd()
    try:
        os.chdir(os.path.join(main_folder, f"models/Random_{nbs_point_traj-1}/TC_{transaction_cost[0]*100}"))

        auto_name = f"auto{str(nbs_latent_spc)}_{str(lambda_auto)}" if autoencoder==True else "noauto"
        transaction_cost_name = f"TC_{str('_'.join(map(str, [x * 100 for x in transaction_cost])))}" if sum(transaction_cost)!= 0 else str(0)
        rebalance_name = f"learned_reebalance_integral"

        if loss_type == "CVaR":
            name = f"Deep_agent_{network}_{auto_name}_dropout_{str(int(dropout_par*100))}_{loss_type}_{str(int(riskaversion*100))}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{lambda_m}"
        else:
            name = f"Deep_agent_{network}_{auto_name}_dropout_{str(int(dropout_par*100))}_{loss_type}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{lambda_m}"

        
        # Compile the neural network
        process = 'training'
        rl_network = DeepAgent(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lambda_m, constraint_min, loss_type, lr, dropout_par, autoencoder, nbs_latent_spc, lambda_auto, name, process)
                             
        print("-------------------------------------------------------------")
        print(name)
        print("-------------------------------------------------------------")

        # Start training
        start = dt.datetime.now()
        print('---Training start---')
        with tf.Session() as sess:
            rl_network.restore(sess, f"{name}.ckpt")
            loss_train_epoch = rl_network.train_deephedging(train_input, underlying_train, underlying_test, HO_train, HO_test, cash_flows_train, cash_flows_test, risk_free_factor, dividendyield_factor, transaction_cost, rebalancing_const, riskaversion, test_input, sess, epochs)
        print('---Training end---')

        if display_plot == True:
            # Plot the learning curve on the train set, i.e. the CVaR on the train set at the end of each epoch
            lin_nb_epoch = np.linspace(1, loss_train_epoch.shape[0], loss_train_epoch.shape[0])
            fig = plt.figure(figsize=(5, 5), dpi=100)
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,0],label="Train error")
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,1],label="Test error")
            plt.title(f"Loss function: {loss_type}")
            plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            plt.show()
    finally:
        #change dir back to original working directory (owd)
        os.chdir(owd)
    
    return loss_train_epoch


def network_inference(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lambda_m, constraint_min,
                   loss_type, lr, dropout_par, train_input, underlying_train, underlying_test, HO_train, HO_test,
                     cash_flows_train, cash_flows_test, risk_free_factor,dividendyield_factor, transaction_cost,
                       riskaversion, test_input, epochs, display_plot, id, autoencoder, nbs_latent_spc, lambda_auto, rebalancing_const):
    
    # Function to test deep hedging algorithm
    """
    Input:
    network              : Neural network architecture {"LSTM","RNNFNN","FFNN"}
    nbs_point_traj       : time steps 
    batch_size           : batch size {296,1000}
    nbs_input            : number of features
    nbs_units            : neurons per layer/cell
    nbs_assets           : number of hedging intruments
    constraint_min       : minimum value of the portfolio for soft constraint
    lambda_m             : labmda for lagrange multiplier (soft constraint)
    autoencoder          : autoencoder component included
    nbs_latent_spc       : dimension of the latent space
    lambda_auto          : regularization parameter to ponderate autoencoder loss function
    loss_type            : loss function {"CVaR","MSE","SMSE"}
    lr                   : learning rate of the Adam optimizer
    dropout_par          : dropout regularization parameter 
    transaction_cost     : Proportional transaction cost [0,5/10000,5/1000,1/100]
    riskaversion         : CVaR confidence level (0,1)
    epochs               : Number of epochs, training iterations
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
    display_plot         : Display plot of training and validation loss

    Output:
    deltas           : Position in the hedging instruments

    """

    
    owd = os.getcwd()

    os.chdir(os.path.join(main_folder, f"models/Random_{nbs_point_traj-1}/TC_{transaction_cost[0]*100}"))

    auto_name = f"auto{str(nbs_latent_spc)}_{str(lambda_auto)}" if autoencoder==True else "noauto"
    transaction_cost_name = f"TC_{str('_'.join(map(str, [x * 100 for x in transaction_cost])))}" if sum(transaction_cost)!= 0 else str(0)
    rebalance_name = f"learned_reebalance_integral"

    if loss_type == "CVaR":
        name = f"Deep_agent_{network}_{auto_name}_dropout_{str(int(dropout_par*100))}_{loss_type}_{str(int(riskaversion*100))}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{lambda_m}"
    else:
        name = f"Deep_agent_{network}_{auto_name}_dropout_{str(int(dropout_par*100))}_{loss_type}_{transaction_cost_name}_{rebalance_name}_{id}_softcons_{lambda_m}"


    # Compile the neural network
    process = 'inference'
    rl_network = DeepAgent(network, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, lambda_m, constraint_min, loss_type, lr, dropout_par, autoencoder, nbs_latent_spc, lambda_auto, name, process)
                            
    print("-------------------------------------------------------------")
    print(name)
    print("-------------------------------------------------------------")

    # Start training
    start = dt.datetime.now()
    print('---Inference start---')
    with tf.Session() as sess:
        rl_network.restore(sess, f"{name}.ckpt")
        deltas, latent, portfolio, constraint = rl_network.predict(test_input, underlying_test, HO_test, cash_flows_test, risk_free_factor, dividendyield_factor, transaction_cost, rebalancing_const, riskaversion, sess)
        os.chdir(owd)
        backtest = False
        if backtest==True:
            os.chdir(os.path.join(main_folder, f"data/results/Backtest/Random_{nbs_point_traj-1}/TC_{transaction_cost[0]*100}"))
        else:
            os.chdir(os.path.join(main_folder, f"data/results/Training/Random_{nbs_point_traj-1}/TC_{transaction_cost[0]*100}"))
        
        np.save(f"{name}",deltas)
        np.save(f"latent_{name}",latent)
        np.save(f"constraint_{name}",constraint)

    print('---Inference end---')
    os.chdir(owd)

    return deltas,latent, portfolio, constraint, name