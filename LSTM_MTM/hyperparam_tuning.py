### LSTM hyperparameter tuning
### Author: Juan Pablo Valdes
### Code adapted from Fuyue Liang LSTM for stirred vessels
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import modeltrain_LSTM as trn
from modeltrain_LSTM import LSTM_S2S, LSTM_DMS
from tools_modeltraining import custom_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.optim as optim
import torch.utils.data as data
import pickle
import psutil
import os
from functools import partial

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


## Env. variables ##

# fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
# input_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/input_data/'
# trainedmod_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/trained_models/'
# tuning_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/tuning'

fig_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/figs/'
input_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/input_data/'
trainedmod_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/trained_models/'
tuningmod_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/tuning/'

########################################### METHODS ###########################################

def train_tune(config, model_choice, init, X_tens, y_tens):
    '''
    init: Initialization (non-tunable) parameters for LSTM class
    config: receives the hyperparameters we would like to train with;
    '''     
    ## Dataloader and loss fun
    loss_fn = custom_loss(config['penalty_weight'])
    
    trainloader = data.DataLoader(data.TensorDataset(X_tens[0], y_tens[0]), 
                                  shuffle=True, batch_size=config['batch_size'])
    valloader = data.DataLoader(data.TensorDataset(X_tens[1], y_tens[1]), 
                                shuffle=True, batch_size=config['batch_size'])


    ## Calling model class instance and training function
    if model_choice == "DMS":

        model = LSTM_DMS(init["input_size"],config['hidden_size'],
                         init["output_size"],init["pred_steps"],
                         config["l1_lambda"], config["l2_lambda"])
        
        optimizer = optim.Adam(model.parameters(), lr = config["learning_rate"])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        ## Calling training function
        trn.train_DMS(model, optimizer, loss_fn, trainloader, valloader, scheduler, 
            init["num_epochs"], init["check_epochs"], X_tens[0], y_tens[0], X_tens[1], 
            y_tens[1],saveas='DMS_out',batch_loss=config["batch_loss"],tuning=True)
        

    elif model_choice == 'S2S':

        model = LSTM_S2S(init["input_size"],config['hidden_size'],
                         init["output_size"],init["pred_steps"],
                         config["l1_lambda"], config["l2_lambda"])
        
        optimizer = optim.Adam(model.parameters(), lr = config["learning_rate"])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
                
        trn.train_S2S(model,optimizer, loss_fn, trainloader, valloader, scheduler, init["num_epochs"], 
                  init["check_epochs"],init["pred_steps"],X_tens[0], y_tens[0], X_tens[1], y_tens[1],
                  config["tf_ratio"], config["dynamic_tf"], config["training_prediction"],
                  saveas='S2S_out',batch_loss=config["batch_loss"],tuning=True)

    else:
        raise ValueError('Model selected is not configured/does not exist. Double check input.')


def load_data(model_choice):
    ##### LOADING DATA ######

    # Initializing loading containers
    set_labels = ["train", "val", "test"]
    # Will contain training and validation sets in positions 0,1.
    windowed_in_tens = []
    windowed_out_tens = []
    in_casebatch = []
    out_casebatch = []

    test_array = []
    testset_labels = []

    ## Loading test_numpy array
    npfile = os.path.join(trainedmod_savepath,f'data_sets_{model_choice}', f'{set_labels[-1]}_pkg.pkl')

    with open(npfile, 'rb') as file:
        test_pkg = pickle.load(file)

    test_array.append(test_pkg[f"{set_labels[-1]}_arr"])
    testset_labels.append(test_pkg["splitset_labels"])

    ## Loading training and validation windowed tensors
    for setlbl in set_labels:
        in_ptfile = os.path.join(trainedmod_savepath,f'data_sets_{model_choice}', f'X_{setlbl}.pt')
        out_ptfile = os.path.join(trainedmod_savepath,f'data_sets_{model_choice}', f'y_{setlbl}.pt')

        if os.path.exists(in_ptfile) and os.path.exists(out_ptfile):

            # X_tensors
            in_savetens = torch.load(in_ptfile)
            windowed_in_tens.append(in_savetens["windowed_data"])
            in_casebatch.append(in_savetens[f"{setlbl}_casebatch"])

            # y_tensors
            out_savetens = torch.load(out_ptfile)
            windowed_out_tens.append(out_savetens["windowed_data"])
            out_casebatch.append(out_savetens[f"{setlbl}_casebatch"])
    
    return windowed_in_tens, windowed_out_tens, in_casebatch, out_casebatch, test_array, testset_labels


########################################### MAIN ###########################################

def main():

    model_choice = input('Select a LSTM model to tune (DMS, S2S): ')
    
    ## Load windowed tensors for training and val
    X_tens, y_tens, _, _ ,_, _ = load_data(model_choice)

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
        'l1_lambda' : tune.loguniform(0.0001, 0.1),
        'l2_lambda' : tune.loguniform(0.0001, 0.1),
        'batch_loss' : tune.choice(['True', 'False']),
        'penalty_weight' : tune.choice([0.6,0.7,0.8,0.9])
    }
    
    init = {
        "input_size": X_tens[0].shape[-1],
        "output_size": y_tens[0].shape[-1],
        "pred_steps": 15,
        "num_epochs": 50,
        "check_epochs": 10
    }

    scheduler = ASHAScheduler(
    metric='val_loss',
    mode='min',
    max_t= init["num_epochs"],
    grace_period=20, # save period without early stopping
    reduction_factor=2,
    )

    ray.shutdown()
    ray.init(num_cpus=num_cpus_to_allocate)

    tuner = tune.run(
    partial(train_tune,model_choice=model_choice,
            init=init,X_tens=X_tens,y_tens=y_tens),
    config = search_space,
    num_samples = 5, # number of hyperparameter configuration to try
    scheduler=scheduler,
    local_dir = os.path.join(tuningmod_savepath,model_choice)
)
    
    best_trial = tuner.get_best_trial('val_loss', 'min', 'last')

    print(f'Best trial id: {best_trial.trial_id}')
    print(f'Best trial config: {best_trial.config}')


if __name__ == "__main__":
    main()