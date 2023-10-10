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
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau


## Env. variables ##

fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
input_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/input_data/'
trainedmod_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/trained_models/'

#fig_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/figs/'
#input_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX//LSTM_MTM/input_data/'
#trainedmod_savepath = ''

##################################### CLASSES #################################################

class Window_data():

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

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            color_palette = color_palettes[label]

            ## Looping per feature number in each split set
            for i in range(data.shape[-1]):

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
    def window_data(self, df, steps_in, stride, steps_out):
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
            for j in range(0, df_case.shape[0]-window_size+1, stride): # Looping over number of rows/windows, depending on window size and number of timesteps (df.shape[0])
                wd_data = df_case[j:j+window_size] # window with times: steps_in + steps_out: entire row
                X.append(wd_data[:-steps_out]) #input values, steps_in
                y.append(wd_data[-steps_out:]) #training/ prediction values, steps_out
            casebatch_lens.append(len(X)) # appending casebatch length per case 

        ## number of windows/rows with size (steps_in) per case, used to later plot per case
        print(casebatch_lens)

        X_array = np.array(X)
        y_array = np.array(y)
        
        return torch.tensor(X_array), torch.tensor(y_array), np.array(casebatch_lens)

class LSTM_DMS(nn.Module):
    
    ## class constructor
    def __init__(self, input_size, hidden_size, output_size, pred_steps,
                 l1_lambda=0.0, l2_lambda=0.0):
        
        # calling the constructor of the parent class nn.Module to properly intialize this class
        super(LSTM_DMS,self).__init__()

        #LSTM attributes
        self.hidden_size = hidden_size
        self.pred_steps = pred_steps # prediction steps = steps_out
        self.output_size = output_size # number of features per output step.
        ## LSTM unit/cell instance from parent class
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) 
        # Linear/dense layer instance from parent class, for decoding multi-step predictions
        self.linear = nn.Linear(hidden_size, output_size * pred_steps)

        # Relevance markers for L1 and L2 regularizations
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    ### forward pass: How input data will be processed by the network layers
    def forward(self, input):

        # No initialisation for hidden or cell states h0, c0. 
        # Inputting data (x_set) into the LSTM cell sequence and reading the output per unit/cell and as a whole at the end
        lstm_output, _ = self.lstm(input)#,(h0,c0)) #shape as (batch_size, input_steps, hidden states)
        
        # Get the hidden state from the last input step given to the LSTM sequence
        last_output = lstm_output[:, -1, :]
        
        # Input the last output from the LSTM sequence into the dense linear layer, where we obtain the multi-output
        multi_step_output = self.linear(last_output)
        
        # Reshape the output to get predictions for multiple future time steps
        multi_step_output = multi_step_output.view(-1, self.pred_steps, self.output_size)

        return multi_step_output
    
    ### Regularization functions to prevent overfitting
    #L1 (lasso) encourages sparse weights
    def l1_regularization_loss(self):
        if self.training:
            l1_loss = 0.0
            for param in self.parameters():
                l1_loss += torch.sum(torch.abs(param))
            return self.l1_lambda * l1_loss
        else:
            return 0

    #L2 (Ridge) encourages small weights
    def l2_regularization_loss(self):
        if self.training:
            l2_loss = 0.0
            for param in self.parameters():
                l2_loss += torch.sum(param ** 2)
            return 0.5 * self.l2_lambda * l2_loss
        else:
            return 0

class custom_loss(nn.Module):

    # constructor and super from the parent nn.Module class
    def __init__(self, penalty_weight):
        super().__init__()
        self.penalty_weight = penalty_weight
        
    def forward(self, prediction, target):
        # mse as base loss function
        mse_loss = nn.MSELoss()(prediction, target)
        
        # penalise negative prediction
        '''
        -prediction: negates all the values in prediction tensor
        torch.relu: set all the now negative values (i.e., initially positive) as zero and postive (previously negative) values unchanged
        torch.mean: average the penalty for all the previously negative predictions
        '''
        penalty = torch.mean(torch.relu(-prediction))
        
        custom_loss = mse_loss + self.penalty_weight * penalty
        
        return custom_loss 

class EarlyStopping:
    def __init__(self, model_name, patience=5, verbose=False, delta=0.0001):
        """
        Args:
            patience (int): How long to wait after last improvement in the monitored metric.
                            Default: 5
            verbose (bool): If True, print a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored metric to be considered an improvement.
                            Default: 0.0001
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.name = model_name

    def __call__(self, val_loss, model):
        """
        Args:
            val_loss (float): Validation loss to be monitored for improvement.
            model (nn.Module): Model to be saved if the monitored metric improves.
        """
        score = -val_loss  # Assuming lower validation loss is better.

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), os.path.join(trainedmod_savepath,f'{self.name}_trained_model.pt'))  # Save the model's state_dict.

####################################### TRAINING FUN. #####################################

def train_DMS(model, optimizer, loss_fn, loader, scheduler,
                  num_epochs, check_epochs, 
                  X_train, y_train, X_val, y_val, saveas):
    
    with open(str(saveas)+'.txt', 'w') as f:
        print(model, file=f)

        ### Early stopping feature to avoid overfitting during training, monitoring a minimum improvement threshold
        early_stopping = EarlyStopping('DMS',patience=10, verbose=True)

        for epoch in range(num_epochs): #looping through epochs
            model.train() #set the model to train mode -- informing features to behave accordingly for training
            
            first_iteration = True
            for X_batch, y_batch in loader:
                
                optimizer.zero_grad() # setting gradients to zero to start a new run on weight optimisation (clear accumulated from previous batch)

                # Forward pass
                y_pred = model(X_batch)

                # Calculate loss
                loss = loss_fn(y_pred, y_batch)

                # Calculate L1 and L2 regularization terms
                l1_regularization = model.l1_regularization_loss()
                l2_regularization = model.l2_regularization_loss()

                # Add regularization terms to the loss
                loss += l1_regularization + l2_regularization

                # Backpropagation and parameter update
                loss.backward() # calculating the gradient of the loss with respect to the model's parameters (weights and biases)
                                # it acculmulates the gradients each time we go through the nested loop

                optimizer.step() # updating parameters to minimize the loss function

                # Check the shapes in the first iteration of the first epoch
                if epoch == 0 and first_iteration:
                    print('Input shape:', X_batch.shape)
                    print('Output shape:', y_pred.shape)
                    first_iteration = False

            # Validation at each check epoch batch
            if epoch % check_epochs != 0:
                continue

            model.eval() # set the model to evaluation form

            with torch.no_grad(): #predictions performed with no gradient calculations
                y_pred_train = model(X_train)
                y_pred_val = model(X_val)

                t_rmse = loss_fn(y_pred_train, y_train)
                v_rmse = loss_fn(y_pred_val, y_val)

                print('Epoch %d : train RMSE  %.4f, val RMSE %.4f ' % (epoch, t_rmse, v_rmse), file=f)
                print('Epoch %d : train RMSE  %.4f, val RMSE %.4f ' % (epoch, t_rmse, v_rmse))
                
            ## Learning rate scheduler step
            scheduler.step(v_rmse)

            ## early stopping check to avoid overfitting
            early_stopping(v_rmse, model)

            if early_stopping.early_stop:
                print('Early stopping')
                break

########################################### MAIN ###########################################

def main():

    ### WINDOW DATA ###

    ## Class instance declarations:
    windowing = Window_data()

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

    train_arr, val_arr, test_arr, splitset_labels = windowing.split_cases(
        input_df, train_frac, test_frac, Allcases)
    
    ## plotting split data
    plot_choice = input('plot split data sets? (y/n) :')
    if plot_choice.lower() == 'y' or plot_choice.lower() == 'yes':
        windowing.plot_split_cases(input_df, splitset_labels, train_arr, val_arr, test_arr, 
                            features,Allcases)
    else:
        pass
    
    ## Windowing hyperparameters
    steps_in, steps_out = 36, 20
    stride = 1
    
    #Windowed training data
    X_train, y_train, train_casebatch = windowing.window_data(train_arr, steps_in, stride, steps_out)
    #Windowed validation data
    X_val, y_val, val_casebatch = windowing.window_data(val_arr, steps_in, stride, steps_out)

    print(f"Windowed input training data shape: {X_train.shape}")
    print(f"Training windowed output shape: {y_train.shape}")
    print(f"Windowed input validation data shape: {X_val.shape}")
    print(f"Validation windowed output shape: {y_val.shape}")

    ### LSTM MODEL TRAINING ###

    # Define hyperparameters
    input_size = X_train.shape[-1]  # Number of features in the input tensor
    hidden_size = 128  # Number of hidden units in the LSTM cell, determines how many weights will be used in the hidden state calculations
    num_layers = 1 # Number of LSTM layers per unit
    output_size = y_train.shape[-1]  # Number of output features, same as input in this case
    pred_steps = steps_out # Number of future steps to predict
    batch_size = 10 # How many windows are being processed per pass through the LSTM
    learning_rate = 0.01
    num_epochs = 3000
    check_epochs = 100

    # customize loss function 
    penalty_weight = 10
    loss_fn = custom_loss(penalty_weight)
        
    ## Calling model class instance and training function
    model_choice = input('Select a LSTM model to train (DMS, S2S): ')

    if model_choice == 'DMS':
        # LSTM model instance
        model = LSTM_DMS(input_size, hidden_size, output_size, pred_steps,
                            l1_lambda=0.00, l2_lambda=0.00)
        
        loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate) # optimizer to estimate weights and biases (backpropagation)
            
        # Learning rate scheduler, set on min mode to decrease by factor when validation loss stops decreasing                                       
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        train_DMS(model=model, optimizer=optimizer, loss_fn=loss_fn, loader=loader, scheduler=scheduler, 
            num_epochs=num_epochs, check_epochs=check_epochs,
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            saveas='DMS_out')
        
    elif model_choice == 'S2S':
        # LSTM model instance
        model = LSTM_DMS(input_size, hidden_size, output_size, pred_steps,
                        l1_lambda=0.00, l2_lambda=0.00)
        
    else:
        raise ValueError('Model selected is not configured/does not exist. Double check input.')

    ### SAVING ALL RELEVANT DATA ###

    set_labels = ["train", "val", "test"]
    arrays = [train_arr, val_arr, test_arr]
    windowed_tensors = [X_train, X_val]
    casebatches = [train_casebatch,val_casebatch]

    ## saving train, validation and test data sets previously split and used as input for windowing process, with corresponding labels
    for setlbl, arr, caselbl_list in zip(set_labels, arrays, splitset_labels):

        save_dict = {
        f"{setlbl}_arr": arr,
        "splitset_labels": caselbl_list
    }
        with open(os.path.join(trainedmod_savepath,f'data_sets_{model_choice}', f'{setlbl}_pkg.pkl'), 'wb') as file:
            pickle.dump(save_dict, file)

        print(f"Saved split set data and labels {setlbl}_pkg.pkl")

    ## saving windowed train and validation datasets (pytorch tensors), with corresponding casebatch lengths  
    for setlbl, tens, csbatch in zip(set_labels, windowed_tensors, casebatches):
        
        save_dict = {
        "windowed_data": tens,
        f"{setlbl}_casebatch": csbatch
    }
        file = os.path.join(trainedmod_savepath,f'data_sets_{model_choice}', f'X_{setlbl}.pt')

        torch.save(save_dict, file)

        print(f"Saved torch package X_{setlbl}.pt")
    
    ## save hyperparameters used for model trained for later plotting and rollout prediction
    hyperparams = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "pred_steps": pred_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "steps_in": steps_in,
        "steps_out": steps_out
    }

    with open(os.path.join(trainedmod_savepath,f'hyperparams_{model_choice}.txt'), "w") as file:

        for key, value in hyperparams.items():
            file.write(f"{key}: {value}\n")
    


if __name__ == "__main__":
    main()