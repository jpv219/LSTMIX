### LSTM hyperparameter tuning
### Author: Juan Pablo Valdes and Fuyue Liang
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import modeltrain_LSTM as trn
from modeltrain_LSTM import LSTM_ED, LSTM_FC, GRU_FC, GRU_ED
from tools_modeltraining import custom_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.optim as optim
import torch.utils.data as data
import pickle
import psutil
import shutil
import os
from functools import partial
import sys
from contextlib import redirect_stdout
import time
import tracemalloc
from memory_profiler import profile
from functools import wraps

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as raypickle


## Env. variables ##

#fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
#input_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/input_data/'
#trainedmod_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/trained_models/'
#tuning_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/tuning'

fig_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/figs/'
input_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/input_data/'
trainedmod_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/trained_models/'
tuningmod_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/tuning/'

#fig_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/figs/'
#input_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX//LSTM_MTM/input_data/'
#trainedmod_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/trained_models'
#tuningmod_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/tuning/'

##################################### DECORATORS #################################################

# Custom memory profile decorator
def mem_profile(folder,model):

    if folder == 'tuning':
        file_path = os.path.join(tuningmod_savepath, model,'logs', f"{model}_tuning_memlog.txt")
    else:
        file_path = os.path.join(trainedmod_savepath, f'{model}_logs', f"{model}_furthertrain_memlog.txt")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            return profile(precision=4,stream=open(file_path,'w'))(func)(*args, **kwargs)
        return wrapper
    return decorator

########################################### METHODS ###########################################

def train_tune(config, model_choice, init, X_tens, y_tens, best_chkpt_path, tuning):
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
    
    arch_choice = model_choice.split('_')[-1]

    net_choice = model_choice.split('_')[0]
    
    ## Calling model class instance and training function based on user input selection

    if arch_choice == 'FC':

        if net_choice == 'LSTM':

            # LSTM model instance
            model = LSTM_FC(init["input_size"],config['hidden_size'],
                         init["output_size"],init["pred_steps"],
                         config["l1_lambda"], config["l2_lambda"])
            
        elif net_choice == 'GRU':

            # GRU model instance
            model = GRU_FC(init["input_size"],config['hidden_size'],
                         init["output_size"],init["pred_steps"],
                         config["l1_lambda"], config["l2_lambda"])
        
        optimizer = optim.Adam(model.parameters(), lr = config["learning_rate"])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        if not tuning:
            with open(os.path.join(best_chkpt_path, 'chk_dict.pkl'),'rb') as fp:
                loaded_checkpoint_state = raypickle.load(fp)
                
                model.load_state_dict(loaded_checkpoint_state['model_state_dict'])
                optimizer.load_state_dict(loaded_checkpoint_state['optimizer_state_dict'])

        ## Calling training function
        trn.train_FC(model_choice, model, optimizer, loss_fn, trainloader, valloader, scheduler, 
            init["num_epochs"], init["check_epochs"], X_tens[0], y_tens[0], X_tens[1], 
            y_tens[1],saveas=f'{model_choice}_out',batch_loss=config["batch_loss"],tuning=tuning)
        

    elif arch_choice == 'ED':

        if net_choice == 'LSTM':

            # LSTM model instance
            model = LSTM_ED(init["input_size"],config['hidden_size'],
                         init["output_size"],init["pred_steps"],
                         config["l1_lambda"], config["l2_lambda"])
        
        elif net_choice == 'GRU':
            model = GRU_ED(init["input_size"],config['hidden_size'],
                         init["output_size"],init["pred_steps"],
                         config["l1_lambda"], config["l2_lambda"])
        
        optimizer = optim.Adam(model.parameters(), lr = config["learning_rate"])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        if not tuning:
            with open(os.path.join(best_chkpt_path, 'chk_dict.pkl'),'rb') as fp:
                loaded_checkpoint_state = raypickle.load(fp)
                
                model.load_state_dict(loaded_checkpoint_state['model_state_dict'])
                optimizer.load_state_dict(loaded_checkpoint_state['optimizer_state_dict'])
                
        trn.train_ED(model_choice, model,optimizer, loss_fn, trainloader, valloader, scheduler, init["num_epochs"], 
                  init["check_epochs"],init["pred_steps"],X_tens[0], y_tens[0], X_tens[1], y_tens[1],
                  config["tf_ratio"], config["dynamic_tf"], config["training_prediction"],
                  saveas=f'{model_choice}_out',batch_loss=config["batch_loss"],tuning=tuning)

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


def run_tuning(config, model_choice, init, X_tens, y_tens, scheduler, 
               num_samples, log_file_path, best_chkpt_path, tuning):
    
    with open(log_file_path, 'w',encoding='utf-8', errors='ignore') as f, redirect_stdout(f):

        try:
            tuner = tune.run(
                partial(train_tune,
                        model_choice=model_choice,
                        init=init,
                        X_tens=X_tens,
                        y_tens=y_tens,
                        best_chkpt_path=best_chkpt_path,
                        tuning=tuning),
                config=config,
                num_samples=num_samples,
                scheduler=scheduler,
                local_dir=os.path.join(tuningmod_savepath, model_choice)
            )

            return tuner
        
        finally:
            sys.stdout.flush()


def further_train(model_choice, init_training, X_tens, y_tens, best_trial,best_chkpt):
    
    config_training = best_trial.config

    best_chkpt_path = best_chkpt.path
    
    ## save hyperparameters used for further model trained for later plotting and rollout prediction
    hyperparams = {
        "input_size": init_training['input_size'],
        "hidden_size": config_training['hidden_size'],
        "output_size": init_training['output_size'],
        "pred_steps": init_training['pred_steps'],
        "batch_size": config_training['batch_size'],
        "learning_rate": config_training['learning_rate'],
        "num_epochs": init_training['num_epochs'],
        "check_epochs": init_training['check_epochs'],
        "steps_in": init_training['steps_in'],
        "steps_out": init_training['steps_out'],
        "tf_ratio": config_training['tf_ratio'],
        "dynamic_tf": config_training['dynamic_tf'],
        "penalty_weight": config_training['penalty_weight'],
        "l1" : config_training['l1_lambda'],
        "l2" : config_training['l2_lambda']
    }

    with open(os.path.join(trainedmod_savepath,f'hyperparams_{model_choice}.txt'), "w") as file:

        for key, value in hyperparams.items():
            file.write(f"{key}: {value}\n")

    ## Set to train further mode
    train_tune_dec = mem_profile(folder='training',model=model_choice)(train_tune)
    train_tune_dec(config_training, model_choice, init_training, X_tens, y_tens, best_chkpt_path, tuning=False)


########################################### MAIN ###########################################

def main():

    #Code performance tracking metrics
    start_time = time.time()
    tracemalloc.start()
    
    ## Selection of Neural net architecture and RNN unit type
    net_choice = input('Select a network to use for predictions (GRU/LSTM): ')

    arch_choice = input('Select the specific network architecture (FC/ED): ')

    model_choice = net_choice + '_' + arch_choice
    
    ## Load windowed tensors for training and val
    X_tens, y_tens, _, _ ,_, _ = load_data(model_choice)

    # limit the number of CPU cores used for the whole tuning process
    percent_cpu_to_occupy = 0.3
    total_cpus = psutil.cpu_count(logical=False)
    num_cpus_to_allocate = int(total_cpus * percent_cpu_to_occupy)

    #configure the hyperparameter search spaces for each model to tune
    search_spaces = {
        'FC': {
            'hidden_size': tune.choice([2 ** i for i in range(6, 9)]),
            'learning_rate': tune.choice([0.002,0.005,0.01]),
            'batch_size': tune.choice(range(8, 44, 4)),
            'training_prediction': tune.choice(['none']),
            'tf_ratio': tune.choice([0]),
            'dynamic_tf': tune.choice(['False']),
            'l1_lambda': tune.choice([0, 0.00001]),
            'l2_lambda': tune.choice([0, 0.00001,0.0001]),
            'batch_loss': tune.choice(['False']),
            'penalty_weight': tune.choice([0.01,0.1,1,10])
        },
        'ED': {
            'hidden_size': tune.choice([2 ** i for i in range(6, 9)]),
            'learning_rate': tune.choice([0.002,0.005]),
            'batch_size': tune.choice(range(8, 44, 4)),
            'training_prediction': tune.choice(['recursive','teacher_forcing', 'mixed']),
            'tf_ratio': tune.choice([0.02,0.1, 0.2, 0.4]),
            'dynamic_tf': tune.choice(['True','False']),
            'l1_lambda': tune.choice([0]),
            'l2_lambda': tune.choice([0]),
            'batch_loss': tune.choice(['False']),
            'penalty_weight': tune.choice([0.1])
        }
    }

    search_space = search_spaces[arch_choice]
    
    # Set constant parameters to intialize the RNN
    init = {
        "input_size": X_tens[0].shape[-1],
        "output_size": y_tens[0].shape[-1],
        "pred_steps": 30,
        "num_epochs": 150,
        "check_epochs": 30
    }

    # Configure and run RAY TUNING 
    scheduler = ASHAScheduler(
    metric='val_loss',
    mode='min',
    max_t= init["num_epochs"],
    grace_period=40, # save period without early stopping
    reduction_factor=2,
    )

    ray.shutdown()
    ray.init(num_cpus=num_cpus_to_allocate)
    num_samples = 6
    log_file_path = os.path.join(tuningmod_savepath,model_choice,f'logs/{model_choice}_tune_out.log')

    #Decorate the tuner
    tune_dec = mem_profile(folder='tuning',model=model_choice)(run_tuning)

    # Run the experiment
    tuner = tune_dec(search_space, model_choice, init, X_tens, 
                       y_tens,scheduler, num_samples, log_file_path, best_chkpt_path='', tuning=True)
    
    # Extract results from tuning process
    best_trial = tuner.get_best_trial('val_loss', 'min', 'last')
    best_chkpoint = tuner.get_best_checkpoint(best_trial,'val_loss','min')

    ray.shutdown()

    print(f'Finished tuning hyperparameters with {num_samples} samples')
    print(f'Best trial id: {best_trial.trial_id}')
    print(f'Best trial config: {best_trial.config}')

    # Saving best model and config to external path
    best_model_path = os.path.join(tuningmod_savepath,f'best_models/{model_choice}')

    shutil.copy(f'{best_chkpoint.path}/chk_dict.pkl',best_model_path)

    with open(f'{best_model_path}/config_{model_choice}.pkl', 'wb') as pickle_file:

        pickle.dump(best_trial.config, pickle_file)

    print('Model state and config settings copied to best_model folder')

    #### FURTHER TRAINING WITH TUNED MODEL ###

    train_further = input('Train best tuned trial further? (y/n): ')

    if train_further.lower() == 'y':
    
        ## Setting new init and config parameters for further training of best trial tuned
        init_training = {
            "input_size": X_tens[0].shape[-1],
            "output_size": y_tens[0].shape[-1],
            "pred_steps": 30,
            "num_epochs": 3000,
            "check_epochs": 100,
            "steps_in": 40,
            "steps_out": 30
    }
        
        further_train(model_choice,init_training,X_tens,y_tens,best_trial,best_chkpoint)

    #Reporting code performance
    print(f'Total time consumed for {model_choice} hyperparameter tuning and further training: {(time.time()-start_time)/60} min')

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Memory usage throughout the hyperparameter tuning process {current / 10**6}MB; Peak was {peak / 10**6}MB")


if __name__ == "__main__":
    main()