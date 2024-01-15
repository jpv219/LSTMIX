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

## env. variables 
trainedmod_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/trained_svmodels/'
input_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/input_data/'


########################################### MAIN ###########################################

def main():
    
    ####### WINDOW DATA ########

    ## Windowing hyperparameters
    steps_in, steps_out = 50, 50
    stride = 1

    ## Smoothing parameters
    smoothing_method = 'lowess'
    window_size = 5 # needed for moveavg and savgol
    poly_order = 3 # needed for savgol
    lowess_frac = 0.03 #needed for lowess

    smoothing_params = (window_size,poly_order,lowess_frac)

    ## Re process raw data to swap cases between train, validation and test split sets. Order will depend on Allcases list and train/test fracs.
    choice = input('Re-process raw data sets before windowing? (y/n) : ')

    if choice.lower() == 'y':

        ## Cases to split and features to read from 
        # Allcases = ['bi001', 'bi01', 'b09', 'b06pm', 'b03', 'da01pm', 'da01', 'bi01pm', '3drop',
        # 'coarsepm', 'bi001pm', 'bi1',
        # 'b06', 'b09pm', 'da1', 'da001']

        svcases = ['Bi0001','Bi0004','Bi001','B05','B07','clean','B09','Bi1','Bi0002']

        # Random sampling
        cases = random.sample(svcases,len(svcases))

        features = ['Number of drops', 'Interfacial Area']

        trn.input_data(svcases,features,smoothing_method,smoothing_params)

    # Reading saved re-shaped input data from file
    with open(os.path.join(input_savepath,'svinputdata.pkl'), 'rb') as file:
        input_pkg = pickle.load(file)

    # Reading input data sets and labels previously processed and stored
    input_df = input_pkg['smoothed_data']
    Allcases = input_pkg['case_labels']
    features = input_pkg['features']
    
    ## data splitting for training, validating and testing
    train_frac = 0.7
    test_frac = 0.15

    windowed_data = trn.windowing(steps_in,steps_out,stride,train_frac, test_frac, input_df, Allcases,features)

    model_choice = input('Which model would you like to generate data for? (DMS/S2S): ')

    trn.saving_data(windowed_data,hp={},model_choice=model_choice,save_hp=False)

    print(f'Saved data succesfully in {trainedmod_savepath}/data_sets_{model_choice}')


## Saving data for hyperparam tuning

if __name__ == "__main__":
    main()