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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

## Env. variables ##

# fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
# trainedmod_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/trained_models/'

#fig_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/figs/'
#trainedmod_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/trained_models/'

fig_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/figs/'
trainedmod_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/svtrained_models/'

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

fine_labels = {
    # svcases #
    'Bi0001': r'$Bi=0.001$', 'Bi0002': r'$Bi=0.002$', 'Bi0004': r'$Bi=0.004$', 'Bi001': r'$Bi=0.01$', 'Bi1': r'$Bi=1$',
    'B05': r'$Bi=0.1, \beta=0.5$','B07': r'$Bi=0.1, \beta=0.7$', 'B09': r'$Bi=0.1, \beta=0.9$',
    'clean': r'Clean',
    # smx cases #
    'b03': r'$\beta=0.3$','b06':r'$\beta=0.6$','bi001':r'$Bi=0.01$','bi01':r'$Bi=0.1$','da01': r'$Da=0.1$','da1':r'$Da=1$',
    'b06pm':r'$\beta_{pm}=0.6$,','b09pm':r'$\beta_{pm}=0.9$,','bi001pm':r'$Bi_{pm}=0.01$,',
    'bi1':r'$Bi=1$','bi01pm':r'$Bi=0.1$,','3drop':r'3-Drop',
    'b09':r'$\beta=0.9$','da01pm':r'$Da_{pm}=0.1$, ','da001':r'$Da=0.01$', 'coarsepm':r'Pre-Mix'
}

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
            ## Input size for model prediction changes from initial steps in to steps out, as we take last prediction as new input

            rolled_predictions.append(output.detach().clone())

        # Concatenate all predictions into a single tensor
        rolled_predictions = torch.cat(rolled_predictions, dim=1)               

    return rolled_predictions

####################################### PLOTTING FUN. #####################################

def plot_model_pred(model, model_name,features,set_labels,set,
                    X_data,true_data,wind_size,casebatch_len):

    model.eval()

    # Retrieving final training and validation state from windowed X tensor
    with torch.no_grad():
        y_pred_data = model(X_data)

    num_features = len(features) #Nd and IA
    num_cases = len(set_labels)
    colors = sns.color_palette('muted', num_cases)

    # Loop over features
    for f_idx in range(num_features):

        fig, ax = plt.subplots(figsize=(12, 6))

        if f_idx == 0:        
            axins = inset_axes(ax, width='30%', height='30%', loc='upper center')
            axins.set_ylim(0.3,1.1)
        else:
            axins = inset_axes(ax, width='60%', height='30%', loc='lower center',bbox_to_anchor=(0.3, 0.1, 0.5, 0.9), bbox_transform=ax.transAxes)
            axins.set_ylim(0.7,1.1)
        
        axins.tick_params(bottom=True, top=True, left=True, right=True, axis='both', direction='in', length=5, width=1.5)
        axins.grid(color='k', linestyle=':', linewidth=0.1) 

        s_idx = 0 # first element to choose from each row, per case
        x_range = range(wind_size-31,len(true_data)-30)
        s_idx_l = -1
        x_range_l = range(wind_size-1,len(true_data))
        
        # Loop over cases
        for seq, case in zip(range(num_cases), set_labels):
            # Target plots, true data from CFD
            plot_label = fine_labels.get(case,case)

            p = ax.plot(true_data[:, seq, f_idx], 
                         label=f'Target {plot_label}', color=colors[seq % len(colors)], 
                         linewidth = 3) # true_data has shape [times,cases,features]
            
            axins.plot(true_data[:, seq, f_idx], label=f'Target {plot_label}', color=colors[seq % len(colors)], 
                         linewidth = 3)
            
            plt.setp(ax.spines.values(),linewidth = 1.5)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.5)  # change width
            
            # Plot a given position from each output window generated from the windowed X data - representing final training and validation state 
            # y_pred_data has shape (number of rows for all cases (separated by casebatch), timesteps per row (input and output steps for X and y respectively), features)
            if seq == 0:
                ax.plot(x_range,
                        y_pred_data[:casebatch_len[seq],s_idx,f_idx],'s', markersize=5, #separating each case and selecting a given s_idx from each output window/row
                        markerfacecolor=p[0].get_color(),alpha=0.8, markeredgewidth=1.5, markeredgecolor='k',
                        lw=0.0, label=f'Pred. {plot_label}')
                
                #last element from each window insert plot
                axins.plot(x_range_l,y_pred_data[:casebatch_len[seq],s_idx_l,f_idx], 'o', markersize = 5, 
                        markerfacecolor=p[0].get_color(),alpha=0.8, markeredgewidth=1.5, markeredgecolor='k',
                        lw=0.0, label=f'Pred. {plot_label}')
            else:
                ax.plot(x_range,
                        y_pred_data[casebatch_len[seq-1]:casebatch_len[seq],s_idx,f_idx],'s', markersize=5, 
                        markerfacecolor=p[0].get_color(),alpha=0.8, markeredgewidth=1.5, markeredgecolor='k',
                        lw=0.0, label=f'Pred. {plot_label}')
                
                axins.plot(x_range_l,y_pred_data[casebatch_len[seq-1]:casebatch_len[seq],s_idx_l,f_idx], 'o', markersize = 5, 
                        markerfacecolor=p[0].get_color(),alpha=0.8, markeredgewidth=1.5, markeredgecolor='k',
                        lw=0.0, label=f'Pred. {plot_label}')
                   
        ax.legend()
        axins.set_xlim(wind_size-1, len(true_data)+5)
        ax.set_xlim(0, 110)
        ax.set_ylim(0)
        ax.set_title(f'Prediction with LSTM {model_name} for {features[f_idx]} in {set} set')
        ax.set_xlabel('Time steps')
        ax.set_ylabel(f'Scaled {features[f_idx]}')
        ax.grid(color='k', linestyle=':', linewidth=0.1)
        ax.tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', length=5, width=1.5)

        fig.savefig(os.path.join(fig_savepath, f'Pred_{model_name}_{features[f_idx]}_{set}_set.png'), dpi=150)
        plt.show()

def plot_rollout_pred(rollout_seq, true_data, input_steps, features,set_labels, model_name):

    colors = sns.color_palette("tab10", len(set_labels))

    num_features = len(features)

    for f_idx in range(num_features):
        fig = plt.figure(figsize=(12,6))

        # looping through cases
        for i, case in enumerate(set_labels):

            plot_label = fine_labels.get(case,case)
            ## truedata shaped as (timesteps, cases, features) and rollout as (case,timestep,features)
            r2 = r2_score(true_data[:,i,f_idx],rollout_seq[i,:,f_idx][:true_data.shape[0]])

            p = plt.plot(true_data[:,i,f_idx], label=f'Target {plot_label}, $R^2$:{r2:.4f}',color = colors[i % len(colors)],linewidth = 3)

            # plotting rollout predictions from input steps until length of true data. Rolloutseq has shape (cases,timesteps,features)
            plt.plot(range(input_steps,len(true_data)),rollout_seq[i,input_steps:len(true_data),f_idx],'s',markersize=5, # plotting interval after input steps until true data ends
                     markerfacecolor=p[0].get_color(),alpha=0.8,markeredgewidth=1.5, markeredgecolor='k',
                     lw=0.0, label=f'{model_name} Pred. {plot_label}')
            ax = plt.gca()

            plt.setp(ax.spines.values(),linewidth = 1.5)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.5)  # change width
        
        plt.legend()
        plt.title(f'Rollout pred with LSTM {model_name} for: {features[f_idx]}')
        # plt.xlim(0, 110)
        # plt.ylim(0)
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

    ## Loading numpyarrays for all split datasets and labels, as well as windowed training and val tensors with casebatch lengths
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

    ## plot final training state from train and validation data
    pred_choice = input('plot predicted training and val data? (y/n) :')

    if pred_choice.lower() == 'y' or pred_choice.lower() == 'yes':

        X_train = windowed_tensors[0].to(torch.float32)
        X_val = windowed_tensors[1].to(torch.float32)
        train_arr = arrays[0]
        val_arr = arrays[1]
        train_casebatch = casebatches[0]
        val_casebatch = casebatches[1]

        # using windowed tensors for plots to represent final training and validation state
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

    plot_rollout_pred(rollout_seq,test_arr, hyperparams['steps_in'], features,splitset_labels[2], model_choice)

if __name__ == "__main__":
    main()