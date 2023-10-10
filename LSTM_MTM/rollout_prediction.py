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
from train_LSTM import LSTM_DMS

## Env. variables ##

fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
trainedmod_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/trained_models/'

####################################### ROLLOUT PREDICTION #####################################


####################################### PLOTTING FUN. #####################################

def plot_model_pred(model,features,set_labels,set,
                    X_data,true_data,wind_size,casebatch_len):

    model.eval()

    with torch.no_grad():
        y_pred_data = model(X_data)

    num_features = len(features) #Nd and IA
    num_cases = len(set_labels)
    colors = sns.color_palette("coolwarm", num_cases)

    # Loop over features
    for f_idx in range(num_features):
        fig = plt.figure(figsize=(12,6))
        s_idx = -1
        
        for seq, case in zip(range(num_cases), set_labels):
            # Target plots, true data from CFD
            p = plt.plot(true_data[:, seq, f_idx], label=f'Target {str(case)}', color=colors[seq % len(colors)], linewidth = 2) # true_data has shape [times,cases,features]
            
            # Train predicted values
            if seq == 0:
                plt.plot(range(wind_size-1,len(true_data)),
                        y_pred_data[:casebatch_len[seq],s_idx,f_idx], 
                        c=p[0].get_color(),linestyle=':', label=f'{set} predicted {str(case)}',linewidth = 4)
            else:
                plt.plot(range(wind_size-1,len(true_data)),
                        y_pred_data[casebatch_len[seq-1]:casebatch_len[seq],s_idx,f_idx], 
                        c=p[0].get_color(),linestyle=':', label=f'{set} predicted {str(case)}', linewidth = 4)
                
        if f_idx == 1:
            plt.ylim(0.6, 1.1)


        plt.legend()
        plt.xlim(40, 105)
        plt.title(f'{set} prediction for {features[f_idx]}')
        plt.xlabel('Time steps')
        plt.ylabel(f'Scaled {features[f_idx]}')

        fig.savefig(os.path.join(fig_savepath, f'{set}_{features[f_idx]}.png'), dpi=150)
        plt.show


def main():
    
    features = ['Number of drops', 'Interfacial Area']

    ## Select LSTM model trained to use for predictions and plots
    model_choice = input('Select a LSTM model to use for predictions (DMS, S2S): ')

    # Loading containers
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

    ## LOADING numpyarrays for all split datasets and labels, as well as windowed training and val tensors with casebatch lengths
    for setlbl in set_labels:
        npfile = os.path.join(trainedmod_savepath,f'data_sets_{model_choice}', f'{setlbl}_pkg.pkl')
        ptfile = os.path.join(trainedmod_savepath,f'data_sets_{model_choice}', f'X_{setlbl}.pt')

        ## Loading pkg with numpyarrays with input data before windowing
        with open(npfile, 'rb') as file:
            save_pkg = pickle.load(file)

        arrays.append(save_pkg[f"{setlbl}_arr"])
        splitset_labels.append(save_pkg["splitset_labels"])

        ## Loading windowed training and val datasets as pt tensors
        if os.path.exists(ptfile):
            save_tens = torch.load(ptfile)
            windowed_tensors.append(save_tens["windowed_data"])
            casebatches.append(save_tens[f"{setlbl}_casebatch"])
    
    wind_size = hyperparams["steps_in"] + hyperparams["steps_out"]

    if model_choice == 'DMS':
        model = LSTM_DMS(hyperparams["input_size"], hyperparams["hidden_size"],
                         hyperparams["output_size"], hyperparams["pred_steps"],
                            l1_lambda=0.00, l2_lambda=0.00)
    elif model_choice == 'S2S':
        pass


    ## Load the last best model before training degrades         
    model.load_state_dict(torch.load(os.path.join(trainedmod_savepath,'DMS_trained_model.pt')))

    ## plot final predictions from train and validation data
    pred_choice = input('plot predicted training and val data? (y/n) :')

    if pred_choice.lower() == 'y' or pred_choice.lower() == 'yes':

        X_train = windowed_tensors[0]
        train_arr = arrays[0]
        train_casebatch = casebatches[0]

        plot_model_pred(model, features, splitset_labels[0],
                        'Train',X_train, train_arr, wind_size, train_casebatch)
    else:
        pass

if __name__ == "__main__":
    main()