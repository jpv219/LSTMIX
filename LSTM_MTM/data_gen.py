### Data generation for LSTM
### Authors: Juan Pablo Valdes and Fuyue Liang
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import modeltrain_LSTM as trn
import random
import os
import pickle
# For input preprocessing methods for different mixers
import configparser
import ast

## env. variables 
trainedmod_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/trained_models/'
input_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/input_data/'
raw_datapath = '/home/pv219/Documents/ML/LSTM_SMX/RawData'

#trainedmod_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/trained_models/'
#input_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/input_data/'

#input_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX//LSTM_MTM/input_data/'
#trainedmod_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/trained_models'

########################################### MAIN ###########################################

def main():
    
    # Read the case-specific info from config file
    mixer_choice = input('Choose the mixing system you would like to pre-process (static/stirred): ')
    config = configparser.ConfigParser()
    config.read(os.path.join(raw_datapath,f'config_{mixer_choice}.ini'))

    ####### WINDOW DATA ########

    steps_in, steps_out = config['Windowing']['steps_in'], config['Windowing']['steps_out']
    stride = config['Windowing']['stride']

    ## Smoothing parameters
    smoothing_method = config['Smoothing']['method']
    window_size = config['Smoothing']['window_size']# needed for moveavg and savgol
    poly_order = config['Smoothing']['poly_order']# needed for savgol
    lowess_frac = config['Smoothing']['lowess_frac']#needed for lowess

    smoothing_params = (window_size,poly_order,lowess_frac)

    ## Re process raw data to swap cases between train, validation and test split sets. Order will depend on Allcases list and train/test fracs.
    choice = input('Re-process raw data sets before windowing? (y/n) : ')

    if choice.lower() == 'y':

        ## Cases to split and features to read from 
        Allcases = ast.literal_eval(config.get('Cases', 'cases'))

        # Random sampling
        cases = random.sample(Allcases,len(Allcases))

        # List of features to be normalized (without DSD)
        feature_map = {'Number of drops': 'Nd',
                    'Interfacial Area': 'IA'
                    }
        norm_columns = ['Number of drops', 'Interfacial Area']

        trn.input_data(Allcases,feature_map,norm_columns,smoothing_method,smoothing_params)

    # Reading saved re-shaped input data from file
    with open(os.path.join(input_savepath,'inputdata.pkl'), 'rb') as file:
        input_pkg = pickle.load(file)

    # Reading input data sets and labels previously processed and stored
    input_df = input_pkg['smoothed_data']
    Allcases = input_pkg['case_labels']
    features = input_pkg['features']
    bins = input_pkg.get('bin_edges', None)

    if bins is None:
        bin_edges = []
    else:
        bin_edges = bins
    
    ## data splitting for training, validating and testing
    train_frac = float(config['Splitting']['train_frac'])
    test_frac = float(config['Splitting']['test_frac'])

    windowed_data = trn.windowing(steps_in,steps_out,stride,train_frac, test_frac, input_df, Allcases,features,bin_edges)

    model_choice = input('Which model would you like to generate data for? (LSTM-FC,LSTM-ED,GRU-FC): ')

    trn.saving_data(windowed_data,hp={},model_choice=model_choice,save_hp=False)

    print(f'Saved data succesfully in {trainedmod_savepath}/data_sets_{model_choice}')


## Saving data for hyperparam tuning

if __name__ == "__main__":
    main()