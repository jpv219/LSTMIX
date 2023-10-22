### Data generation for LSTM
### Authors: Juan Pablo Valdes and Fuyue Liang
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import modeltrain_LSTM as trn

## env. variables 
trainedmod_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/trained_models/'

########################################### MAIN ###########################################

def main():
    
    ####### WINDOW DATA ########

    ## Windowing hyperparameters
    steps_in, steps_out = 30, 15
    stride = 1

    ## Cases to split and features to read from 
    Allcases = ['b03','b06','bi001','bi01','da01','da1','b06pm','b09pm','bi001pm',
    'bi1','bi01pm','3drop',
    'b09','da01pm','da001', 'coarsepm']

    features = ['Number of drops', 'Interfacial Area']

    ## Smoothing parameters
    smoothing_method = 'savgol'
    window_size = 5 # needed for moveavg and savgol
    poly_order = 3 # needed for savgol
    lowess_frac = 0.03 #needed for lowess

    smoothing_params = (window_size,poly_order,lowess_frac)

    ## Re process raw data to swap cases between train, validation and test split sets. Order will depend on Allcases list and train/test fracs.
    choice = input('Re-process raw data sets before windowing? (y/n) : ')

    if choice.lower() == 'y':
        trn.input_data(Allcases,features,smoothing_method,smoothing_params)

    ## data splitting for training, validating and testing
    train_frac = 9/16
    test_frac = 4/16

    windowed_data = trn.windowing(steps_in,steps_out,stride,train_frac, test_frac, Allcases,features)

    model_choice = input('Which model would you like to generate data for? (DMS/S2S): ')

    trn.saving_data(windowed_data,hp={},model_choice=model_choice,save_hp=False)

    print(f'Saved data succesfully in {trainedmod_savepath}/data_sets_{model_choice}')


## Saving data for hyperparam tuning

if __name__ == "__main__":
    main()