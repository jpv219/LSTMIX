### LSTM hyperparameter tuning
### Author: Juan Pablo Valdes
### Code adapted from Fuyue Liang LSTM for stirred vessels
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import modeltrain_LSTM as trn
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
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler


## Env. variables ##

fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
input_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/input_data/'
trainedmod_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/trained_models/'
tuning_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/tuning'

########################################### METHODS ###########################################

def train_tune(model_choice, init, config):
    '''
    init: Initialization (non-tunable) parameters for LSTM class
    config: receive the hyperparameters we would like to train with;
    '''

    ## Calling model class instance and training function
    if model_choice == "DMS":
        model = LSTM_DMS(init["input_size"],config['hidden_size'],
                         init["output_size"],init["pred_steps"],
                         config["l1_lambda"], config["l2_lambda"])

    elif model_choice == 'S2S':
        model = LSTM_S2S(init["input_size"],config['hidden_size'],
                         init["output_size"],init["pred_steps"],
                         config["l1_lambda"], config["l2_lambda"])

    else:
        raise ValueError('Model selected is not configured/does not exist. Double check input.')
    
    ##### LOADING DATA ######

    

########################################### MAIN ###########################################

def main():

    # limit the number of CPU cores used for the whole tuning process
    percent_cpu_to_occupy = 0.3
    total_cpus = psutil.cpu_count(logical=False)
    num_cpus_to_allocate = int(total_cpus * percent_cpu_to_occupy)

    #configure the hyperparameter search space
    search_space = {
        'hidden_size': tune.choice([2 ** i for i in range(5,9)]),
        'learning_rate': tune.loguniform(0.0001, 0.1), # uniformly sampled between 0.0001 and 0.1
        'batch_size': tune.choice(range(5,40,5)),
        'training_prediction': tune.choice(['teacher_forcing', 'mixed']),
        'tf_ratio': tune.choice([0.05,0.1,0.3,0.5,0.7]),
        'dynamic_tf': tune.choice(['True', 'False']),
        'l1_lambda' : tune.loguniform(0, 0.1),
        'l2_lambda' : tune.loguniform(0, 0.1),
        'batch_loss' : tune.choice(['True', 'False'])
    }

    scheduler = ASHAScheduler(
    metric='loss',
    mode='min',
    max_t= 60,
    grace_period=20, # save period without early stopping
    reduction_factor=2,
    )

    model_choice = input('Select a LSTM model to tune (DMS, S2S): ')

    loss_fn = custom_loss(penalty_weight=0.8)





if __name__ == "__main__":
    main()