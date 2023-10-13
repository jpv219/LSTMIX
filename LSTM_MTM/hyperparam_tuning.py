### LSTM hyperparameter tuning
### Author: Juan Pablo Valdes
### Code adapted from Fuyue Liang LSTM for stirred vessels
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

from modeltrain_LSTM import LSTM_S2S, LSTM_DMS  
import rollout_prediction
from tools_modeltraining import custom_loss, EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import pickle
import psutil

import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from functools import partial

## Env. variables ##

fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
input_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/input_data/'
trainedmod_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/trained_models/'
tuning_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/tuning'

########################################### METHODS ###########################################

def train_tune(model_choice, init, config, data_dir, loss_fn, ):
    '''
    init: Initialization (non-tunable) parameters for LSTM class
    config: receive the hyperparameters we would like to train with;
    data_dir: specifies the directory where we load and store the data
    '''
    ## Calling model class instance and training function
    if model_choice == "DMS":
        model = LSTM_DMS(init["input_size"],config['hidden_size'],
                         init["output_size"],init["pred_steps"],
                         l1_lambda=0.00, l2_lambda=0.00)

    elif model_choice == 'S2S':
        model = LSTM_S2S(init["input_size"],config['hidden_size'],
                         init["output_size"],init["pred_steps"])

    else:
        raise ValueError('Model selected is not configured/does not exist. Double check input.')
    
    optimizer = optim.Adam(model.parameters(), lr = config['learning_rate'])
    checkpoint = session.get_checkpoint()
    



########################################### MAIN ###########################################

def main():
    
    model_choice = input('Select a LSTM model to tune (DMS, S2S): ')

    loss_fn = custom_loss(penalty_weight=0.8)



    


if __name__ == "__main__":
    main()