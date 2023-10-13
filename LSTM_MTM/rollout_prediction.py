### LSTM rollout prediction
### Author: Juan Pablo Valdes
### Code adapted from Fuyue Liang LSTM for stirred vessels
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import os
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from modeltrain_LSTM import LSTM_DMS, LSTM_S2S
import numpy as np
from sklearn.metrics import r2_score

## Env. variables ##

fig_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/figs/'
trainedmod_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/trained_models/'

## Plot setup

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Computer Modern']})

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 18
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

####################################### ROLLOUT PREDICTION #####################################

# Function to predict future values using rollout with the LSTM model
def rollout(model, input_seq, steps_out,total_steps):
    
    ## setting to eval mode and dropping gradient calculation for prediction
    model.eval()
    with torch.no_grad():
        
        reshaped_input = np.transpose(input_seq, (1,0,2)) #reshaping to case,inputdata,features
        
        tensor_input = torch.Tensor(reshaped_input)

        rolled_predictions = [tensor_input.detach().clone()]

        ### how many predicted windows will be calculated based on the input_steps
        num_forwards = int(total_steps / steps_out) + 1
        print(f'prediction iterates for {num_forwards} times.')

        ## window predicton
        for _ in range(num_forwards):

            output = model(rolled_predictions[-1]) #LSTM pass with latest prediction window as input

            rolled_predictions.append(output.detach().clone())

        # Concatenate all predictions into a single tensor
        rolled_predictions = torch.cat(rolled_predictions, dim=1)               

    return rolled_predictions

####################################### PLOTTING FUN. #####################################

def plot_model_pred(model,model_name,features,set_labels,set,
                    X_data,true_data,wind_size,casebatch_len):

    model.eval()

    with torch.no_grad():
        y_pred_data = model(X_data)

    num_features = len(features) #Nd and IA
    num_cases = len(set_labels)
    colors = sns.color_palette("husl", num_cases)

    # Loop over features
    for f_idx in range(num_features):
        fig = plt.figure(figsize=(12,6))
        s_idx = -1
        
        for seq, case in zip(range(num_cases), set_labels):
            # Target plots, true data from CFD
            p = plt.plot(true_data[:, seq, f_idx], label=f'Target {str(case)}', color=colors[seq % len(colors)], linewidth = 3) # true_data has shape [times,cases,features]
            ax = plt.gca()
            
            plt.setp(ax.spines.values(),linewidth = 1.5)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.5)  # change width
            
            # Train predicted values
            if seq == 0:
                plt.plot(range(wind_size-1,len(true_data)),
                        y_pred_data[:casebatch_len[seq],s_idx,f_idx],'s', markersize=5, 
                        markerfacecolor=p[0].get_color(),alpha=0.8, markeredgewidth=1.5, markeredgecolor='k',
                        lw=0.0, label=f'Pred. {str(case)}')
            else:
                plt.plot(range(wind_size-1,len(true_data)),
                        y_pred_data[casebatch_len[seq-1]:casebatch_len[seq],s_idx,f_idx],'s', markersize=5, 
                        markerfacecolor=p[0].get_color(),alpha=0.8, markeredgewidth=1.5, markeredgecolor='k',
                        lw=0.0, label=f'Pred. {str(case)}')
                
        if f_idx == 1:
            plt.ylim(0.6, 1.1)


        plt.legend()
        plt.xlim(30, 110)
        plt.ylim(0)
        plt.title(f'Prediction with LSTM {model_name} for {features[f_idx]} in {set} set')
        plt.xlabel('Time steps')
        plt.ylabel(f'Scaled {features[f_idx]}')
        plt.grid(color='k', linestyle=':', linewidth=0.1)
        ax.tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', length=5, width=1.5)

        fig.savefig(os.path.join(fig_savepath, f'Pred_{model_name}_{features[f_idx]}_{set}_set.png'), dpi=150)
        plt.show()

def plot_rollout_pred(rollout_seq, true_data, features,set_labels, model_name):

    colors = sns.color_palette("hsv", len(set_labels))

    num_features = len(features)

    for f_idx in range(num_features):
        fig = plt.figure()

        for i, case in enumerate(set_labels):
            ## truedata shaped as (timesteps, cases, features) and rollout as (case,timestep,features)
            r2 = r2_score(true_data[:,i,f_idx],rollout_seq[i,:,f_idx][:true_data.shape[0]])

            p = plt.plot(true_data[:,i,f_idx], label=f'Target {case}, $R^2$:{r2:.4f}',color = colors[i % len(colors)],linewidth = 3)
            plt.plot(rollout_seq[i,:,f_idx],'s',markersize=5,
                     markerfacecolor=p[0].get_color(),alpha=0.8,markeredgewidth=1.5, markeredgecolor='k',
                     lw=0.0, label=f'{model_name} Pred. {case}')
            ax = plt.gca()

            plt.setp(ax.spines.values(),linewidth = 1.5)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.5)  # change width
        
        plt.legend()
        plt.title(f'Rollout pred with LSTM {model_name} for: {features[f_idx]}')
        plt.xlim(30, 110)
        plt.ylim(0)
        plt.xlabel('Time steps')
        plt.ylabel(f'Scaled {features[f_idx]}')
        plt.grid(color='k', linestyle=':', linewidth=0.1)
        plt.tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', length=5, width=1.5)

        fig.savefig(os.path.join(fig_savepath, f'Rollout_{model_name}_{features[f_idx]}.png'), dpi=150)
        plt.show()

def main():
    
    features = ['Number of drops', 'Interfacial Area']

    ## Select LSTM model trained to use for predictions and plots
    model_choice = input('Select a LSTM model to use for predictions (DMS, S2S): ')

    ##### LOADING MODEL, HYPERPARAMS AND DATA ######
    
    # Initializing loading containers
    set_labels = ["train", "val", "test"]
    hyperparams = {}
    arrays = []
    splitset_labels = []
    windowed_tensors = []
    casebatches = []

    ## Loading relevant data saved from training step
    
    ## Hyperparameter loading
    with open(os.path.join(trainedmod_savepath,f'hyperparams_{model_choice}.txt'), "r") as file:
        for line in file:
            key, value = line.strip().split(": ")  # Split each line into key and value
            hyperparams[key] = eval(value)

    ## Lading numpyarrays for all split datasets and labels, as well as windowed training and val tensors with casebatch lengths
    for setlbl in set_labels:
        npfile = os.path.join(trainedmod_savepath,f'data_sets_{model_choice}', f'{setlbl}_pkg.pkl')
        ptfile = os.path.join(trainedmod_savepath,f'data_sets_{model_choice}', f'X_{setlbl}.pt')

        ## Loading pkg with input data as numpyarrays before windowing
        with open(npfile, 'rb') as file:
            save_pkg = pickle.load(file)

        arrays.append(save_pkg[f"{setlbl}_arr"])
        splitset_labels.append(save_pkg["splitset_labels"])

        ## Loading windowed training and val datasets as pt tensors
        if os.path.exists(ptfile):
            save_tens = torch.load(ptfile)
            windowed_tensors.append(save_tens["windowed_data"])
            casebatches.append(save_tens[f"{setlbl}_casebatch"])
    
    ##### PREDICTIONS #####
    
    wind_size = hyperparams["steps_in"] + hyperparams["steps_out"]

    if model_choice == 'DMS':
        model = LSTM_DMS(hyperparams["input_size"], hyperparams["hidden_size"],
                         hyperparams["output_size"], hyperparams["pred_steps"],
                            l1_lambda=0.00, l2_lambda=0.00)
    elif model_choice == 'S2S':
        model = LSTM_S2S(hyperparams["input_size"], hyperparams["hidden_size"],
                         hyperparams["output_size"],hyperparams["pred_steps"])


    ## Load the last best model before training degrades         
    model.load_state_dict(torch.load(os.path.join(trainedmod_savepath,f'{model_choice}_trained_model.pt')))

    ## plot final predictions from train and validation data
    pred_choice = input('plot predicted training and val data? (y/n) :')

    if pred_choice.lower() == 'y' or pred_choice.lower() == 'yes':

        X_train = windowed_tensors[0]
        X_val = windowed_tensors[1]
        train_arr = arrays[0]
        val_arr = arrays[1]
        train_casebatch = casebatches[0]
        val_casebatch = casebatches[1]

        plot_model_pred(model, model_choice, features, splitset_labels[0],
                        'Train',X_train, train_arr, wind_size, train_casebatch)
        
        plot_model_pred(model, model_choice, features, splitset_labels[1],
                'Validation',X_val, val_arr, wind_size, val_casebatch)
    else:
        pass

    ## carry out rollout predictions on testing data sets
    test_arr = arrays[2]

    ## Extracting input steps from test data, keeping all cases and features in shape (in_steps,cases,features)
    input_seq = test_arr[:hyperparams["steps_in"],:,:]

    # Total steps to predict
    total_steps = test_arr.shape[0] - hyperparams["steps_in"]

    ## Calling rollout prediction for test data
    rollout_seq = rollout(model,input_seq,hyperparams["steps_out"],total_steps)

    plot_rollout_pred(rollout_seq,test_arr, features,splitset_labels[2], model_choice)

if __name__ == "__main__":
    main()