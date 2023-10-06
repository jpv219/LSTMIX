### LSTM windowing and model training
### Author: Juan Pablo Valdes
### Code adapted from Fuyue Liang LSTM for stirred vessels
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import torch

### window data ###

## Env. variables ##

fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
input_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/input_data/'

class windowing():

## split cases intro train, test and val data sets
    def split_cases(self, df, train_frac, test_frac, cases):
        '''
        input shape: (times, cases, features)
        
        return train, val data and cases
        '''
        train_size = int(df.shape[1]*train_frac)

        val_size = int(df.shape[1]*(1-test_frac-train_frac))
        
        # split data sets
        train, val, test = df[:, :train_size, :], df[:, train_size:(train_size+val_size), :], df[:,(train_size+val_size):,:]
        print(f'number of train, val and test cases: {train.shape[1]}, {val.shape[1]}, {test.shape[1]}')
        
        ## split cases grouped in three sets, labeled as train, val and test
        train_cases, val_cases , test_cases = cases[:train_size], cases[train_size:(train_size+val_size)], cases[(train_size+val_size):]
        print(f'training cases: {train_cases}, validation cases: {val_cases}, test cases: {test_cases}')
            
        return train, val, test, (train_cases, val_cases, test_cases)

## plot split data sets   
    def plot_split_cases(self, data, splitset_labels, train, val, test, 
                        features, case_labels, dpi=150):

        #Plot setup
        color_palettes = {
        "Training": sns.color_palette("Set1", len(case_labels)),
        "Validation": sns.color_palette("Set2", len(case_labels)),
        "Test": sns.color_palette("Set3", len(case_labels))
    }
        
        rc('text', usetex=True)
        custom_font = {'family': 'serif', 'serif': ['Computer Modern Roman']}
        rc('font', **custom_font)

        train_cases = splitset_labels[0]
        val_cases = splitset_labels[1]
        test_cases = splitset_labels[2]

        ## Looping over all three data sets
        for split_set, label in zip([train, val, test], 
                                    ['Training', 'Validation', 'Test']):
            
            case_labels = train_cases if label == "Training" else val_cases if label == "Validation" else test_cases

            ## Looping per feature number in each split set
            for i in range(data.shape[-1]):
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                color_palette = color_palettes[label]

                for case, idx in zip(case_labels, range(len(case_labels))):
                    ax[i].plot(split_set[:,idx,i],label = f'{str(case)}',color=color_palette[idx % len(color_palette)])
                    ax[i].set_title(f'{label}: {features[i]}')
                    ax[i].set_xlabel('Time steps')
                    ax[i].set_ylabel(f'Scaled {features[i]}')
                    ax[i].legend()

            ## saving figures
            fig.savefig(os.path.join(fig_savepath, f'{label}_{features[i]}.png'), dpi=dpi)

            plt.show()

    ## Generate windows from input data
    def window_data(df, steps_in, stride, steps_out):
        '''
        
        df: with shape (times, cases, features)
        stride: the step size between consecutive windows
        pred_times:(<window_size) predicted future times from current window
        window size: Encompasses both steps_in and steps_out, referring to input seq and prediction seq
        
        lookback period = window_size - steps_out = steps in
        
        '''
        window_size = steps_in + steps_out
        casebatch_lens = [] # List to contain the number of rows/windows per case used for input-->prediction based on the steps_in - steps_out parameters
                            # Can be calculated as: len(timesteps)-window_size+1
        X, y = [], []

        for i in range(df.shape[1]): # looping for each case, df shape of (times, cases, features)
            df_case = df[:,i,:] # df per case
            for j in range(0, len(df_case)-window_size+1, stride): # Looping over timesteps based on the window size
                wd_data = df_case[j:j+window_size] # window with times: steps_in + steps_out
                X.append(wd_data[:-steps_out]) #input values, steps_in
                y.append(wd_data[-steps_out:]) #training/ prediction values, steps_out
            casebatch_lens.append(len(X)) # appending casebatch length per case 

        ## number of windows/rows with size (steps_in) per case, used to later plot per case
        print(casebatch_lens)
        
        return torch.tensor(X), torch.tensor(y), np.array(casebatch_lens)

def main():

    ## Class instance declarations:
    window = windowing()

    Allcases = ['b03','b06','bi001','bi01','da01','da1','b06pm','b09pm','bi001pm',
    'bi1','bi01pm','3drop',
    'b09','da01pm','da001', 'coarsepm']

    features = ['Number of drops', 'Interfacial Area']

    # Reading saved re-shaped input data from file
    with open(os.path.join(input_savepath,'inputdata.pkl'), 'rb') as file:
        input_df = pickle.load(file)
    
    ## data splitting for training, validating and testing
    train_frac = 0.5625
    test_frac = 0.25

    train_df, val_df, test_df, splitset_labels = window.split_cases(
        input_df, train_frac, test_frac, Allcases)
    
    ## plotting split data
    window.plot_split_cases(input_df, splitset_labels, train_df, val_df, test_df, 
                           features,Allcases)
    




if __name__ == "__main__":
    main()