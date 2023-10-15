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
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tools_modeltraining import custom_loss, EarlyStopping


## Env. variables ##

fig_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/figs/'
input_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/input_data/'
trainedmod_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/trained_models/'

#fig_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/figs/'
#input_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX//LSTM_MTM/input_data/'
#trainedmod_savepath = ''

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
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlefine_labels

fine_labels = {
    # svcases #
    'Bi0001': r'$Bi=0.001$', 'Bi0002': r'$Bi=0.002$', 'Bi0004': r'$Bi=0.004$', 'Bi001': r'$Bi=0.01$', 'Bi1': r'$Bi=1$',
    'B05': r'$Bi=0.1, \beta=0.5$','B07': r'$Bi=0.1, \beta=0.7$', 'B09': r'$Bi=0.1, \beta=0.9$',
    'clean': r'Clean',
    # smx cases #
    'b03': r'$\beta=0.3$','b06':r'$\beta=0.6$','bi001':r'$Bi=0.01$','bi01':r'$Bi=0.1$','da01': r'$Da=0.1$','da1':r'$Da=1$',
    'b06pm':r'$\beta=0.6$,','b09pm':r'$\beta=0.9$,','bi001pm':r'$Bi=0.01$,',
    'bi1':r'$Bi=1$','bi01pm':r'$Bi=0.1$,','3drop':r'3-Drop',
    'b09':r'$\beta=0.9$','da01pm':r'$Da=0.1$, ','da001':r'$Da=0.01$', 'coarsepm':r'coarse pm'
}

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
    def plot_split_cases(self, data, fine_labels, splitset_labels, train, val, test, 
                        features, case_labels, dpi=150):

        #Plot setup
        color_palettes = {
        "Training": sns.color_palette("Set1", len(case_labels)),
        "Validation": sns.color_palette("Set2", len(case_labels)),
        "Test": sns.color_palette("Set3", len(case_labels))
    }
        
        train_cases = splitset_labels[0]
        val_cases = splitset_labels[1]
        test_cases = splitset_labels[2]

        ## Looping over all three data sets
        for split_set, label in zip([train, val, test], 
                                    ['Training', 'Validation', 'Test']):
            
            case_labels = train_cases if label == "Training" else val_cases if label == "Validation" else test_cases

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            color_palette = color_palettes[label]
          
            for axis in ax:
                for spine in axis.spines.values():
                    spine.set_linewidth(1.5)

            ## Looping per feature number in each split set
            for i in range(data.shape[-1]):

                for case, idx in zip(case_labels, range(len(case_labels))):
                    plot_label = fine_labels.get(case,case)
                    ax[i].plot(split_set[:,idx,i],label = f'{plot_label}',color=color_palette[idx % len(color_palette)])
                    ax[i].set_title(f'{label}: {features[i]}')
                    ax[i].set_xlabel('Time steps')
                    ax[i].set_ylabel(f'Scaled {features[i]}')
                    ax[i].tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', length=5, width=1.5)
                    ax[i].grid(color='k', linestyle=':', linewidth=0.1)
                    ax[i].legend()

            ## saving figures
            fig.savefig(os.path.join(fig_savepath, f'{label}_data_{features[i]}.png'), dpi=dpi)

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

class LSTM_encoder(nn.Module):

    # Same as LSTM DMS constructor but with no pred_steps or linear layer as encoder feeds decoder LSTM through the hidden states
    def __init__(self,input_size,hidden_size, num_layers=1):
        super(LSTM_encoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True)
        
    # Take the input sequences and output the hidden states for the LSTM decoder section
    def forward(self, encoder_input):
        ''' 
        return encoder_hidden_states: outputs the last time hidden and cell state to be fed into the LSTM decoder
        
        input shape: (batch_size, input steps/input window, input_size=num_features)
        output shape: (input_size=num_features, hidden_size)
        '''
        _, (h_n_encoder,c_n_encoder) = self.lstm(encoder_input) #ignoring output (hidden states) for all times and only saving a tuple with the last timestep cell and hidden state
        
        return (h_n_encoder,c_n_encoder)

class LSTM_decoder(nn.Module):

    ## Same constructor as DMS as now we are decoding the final LSTM cell through a linear layer to generate the final output 
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True)
        
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input, encoder_states):
        '''
        return 
        lstm_output: returns decoded hidden states as output for all times 
        
        input shape: (batch_size, 1, input_size=num_features) the last time step
        output shape: (batch_size, input_size=num_features)
        '''

        # LSTM cell is initialized with the encoder cell and hidden states
                # Input tensor is unsqueezed to introduce an additional dimension in axis = 1 to perform LSTM calculations normally for 1 step
        lstm_output, _ = self.lstm(decoder_input.unsqueeze(1), encoder_states) #Similar to DMS, output is saved, representing all hidden states per timestep
        
        ## output tensor is squeezed, removing the aritificial time dimension in axis = 1, as it will be looped during prediction for each time and appended to a 3D tensor.
        output = self.linear(lstm_output.squeeze(1))
        
        return output
    
class LSTM_S2S(nn.Module):
    ''' Double LSTM Encoder-decoder architecture to make predictions '''

    #Constructing the encoder decoder LSTM architecture
    def __init__(self, input_size, hidden_size, output_size, pred_steps):
        super(LSTM_S2S,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pred_steps = pred_steps #steps out = output window
        
        self.encoder = LSTM_encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = LSTM_decoder(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    def forward(self,input_tensor):
        '''
        input_tensor: shape (batch_size, input steps = input window, input_size=num_features)
        pred_steps: number of time steps to predict
        return np_outputs: array containing predictions
        '''
                
        # encode input_tensor
        encoder_states = self.encoder(input_tensor)

        # initialize output tensor for prediction
        outputs = torch.zeros(input_tensor.shape[0], self.pred_steps, input_tensor.shape[2]) #shape = batch_size, steps_out, num_features


        # decode input_tensor
        decoder_input = input_tensor[:,-1,:] # Taking last value in the window/sequence
        decoder_input_states = encoder_states

        # predictions carried out on the decoder for each time in the output window = steps_out
        for t in range(self.pred_steps):
            decoder_output = self.decoder(decoder_input,decoder_input_states)
            outputs[:,t,:] = decoder_output
            # prediction done recursively
            decoder_input = decoder_output

        np_outputs = outputs.detach().numpy() ## detaching from gradient requirements during prediction

        return torch.from_numpy(np_outputs)


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

def train_S2S(model, optimizer, loss_fn, loader,scheduler, num_epochs, 
              check_epochs, pred_steps, X_train, y_train, X_val, y_val, 
              training_prediction, tf_ratio, dynamic_tf,saveas):
    ''' 
    training_prediction: ('recursive'/'teacher_forcing'/'mixed')
    tf_ratio: float[0,1] 
                relevance on teacher forcing when training_prediction = 'teacher_forcing'.
                For each batch, a random number is generated. 
                If the number is less than tf_ratio, tf is used; otherwise, prediction is done recursively.
                If tf_ratio = 1, only tf is used.
    dynamic_tf: (True/False)
                dynamic teacher forcing reduces the amount of teacher forcing for each epoch
    
    return loss: array of loss function for each epoch
    '''

    # save the training model
    with open(str(saveas)+'.txt', 'w') as f:
        print(model, file=f)

        ### Early stopping feature to avoid overfitting during training, monitoring a minimum improvement threshold
        early_stop = EarlyStopping('S2S',patience=10, verbose=True)

        for epoch in range(num_epochs): #looping through training epochs
            
            model.train() #setting model to training function to deactivate regularization and other training features
            first_iteration = True

            for X_batch, y_batch in loader:

                # initializing output tensor
                outputs = torch.zeros(X_batch.shape[0], pred_steps, X_batch.shape[2]) #shape = (batch_size,steps_out,num_features)

                #reset gradients from previous training step
                optimizer.zero_grad()

                #going through the LSTM encoder layer: return hidden and cell states
                encoder_states = model.encoder(X_batch)

                # decoder starting with teacher forcing: input set as last timestep from input batch
                decoder_input = X_batch[:,-1,:] # in shape of (batch_size, input_size = num_features)
                decoder_input_states = encoder_states

                #Considering variations in training methods per batch
                if training_prediction == 'recursive':
                        
                    # recursive prediction: predicted output is fed
                        for t in range(pred_steps):
                            decoder_output = model.decoder(decoder_input, decoder_input_states)
                            outputs[:,t,:] = decoder_output
                            decoder_input = decoder_output


                if training_prediction == 'teacher_forcing':
                        
                    # predict using teacher forcing: target is fed
                        if random.random() < tf_ratio:
                            for t in range(pred_steps):
                                decoder_output = model.decoder(decoder_input, decoder_input_states)
                                outputs[:,t,:] = decoder_output
                                decoder_input = y_batch[:,t,:] # target fed from y_batch in shape of (batch_size, input_size = num_features)
                        # predict recursively
                        else:
                            for t in range(pred_steps):
                                decoder_output = model.decoder(decoder_input, decoder_input_states)
                                outputs[:,t,:] = decoder_output
                                decoder_input = decoder_output


                if training_prediction == 'mixed':

                    # both types of training methods used in the same batch, alternating stochastically based on tf_ratio
                    for t in range(pred_steps):
                        decoder_output = model.decoder(decoder_input, decoder_input_states)
                        outputs[:,t,:] = decoder_output

                        ## Teaching method chosen per timestep within the given batch
                        # teacher forcing
                        if random.random() < tf_ratio:
                            decoder_input = y_batch[:,t,:]
                        # recursive:
                        else:
                            decoder_input = decoder_output

                loss = loss_fn(outputs,y_batch)

                # Backpropagation and parameter update
                loss.backward() # calculating the gradient of the loss with respect to the model's parameters (weights and biases)
                                # it acculmulates the gradients each time we go through the nested loop

                optimizer.step() # updating parameters to minimize the loss function

                # Check the shapes in the first iteration of the first epoch
                if epoch == 0 and first_iteration:
                    print('Input shape:', X_batch.shape)
                    print('Output shape:', outputs.shape)
                    first_iteration = False
                
            # dynamic teacher forcing
            if dynamic_tf and tf_ratio > 0:
                tf_ratio = tf_ratio - 0.02 ## if dynamic tf active, the amount of teacher forcing is reduced per epoch

                # Validation at each check epoch batch
            if epoch % check_epochs != 0:
                continue

            model.eval() # set the model to evaluation form, disabling regularisation and training features

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
            early_stop(v_rmse, model)

            if early_stop.early_stop:
                print('Early stopping')
                break

########################################### MAIN ###########################################

def main():

    ####### WINDOW DATA ########

    ## Class instance declarations:
    windowing = Window_data()

    # Allcases = ['b03','b06','bi001','bi01','da01','da1','b06pm','b09pm','bi001pm',
    # 'bi1','bi01pm','3drop',
    # 'b09','da01pm','da001', 'coarsepm']

    svcases = ['Bi0001','Bi0002','Bi0004','Bi001','B07','clean','B09', 'B05', 'Bi1']

    features = ['Number of drops', 'Interfacial Area']

    # Reading saved re-shaped input data from file
    with open(os.path.join(input_savepath,'svinputdata.pkl'), 'rb') as file:
        input_df = pickle.load(file)
    
    ## data splitting for training, validating and testing
    train_frac = 0.7
    test_frac = 0.15

    train_arr, val_arr, test_arr, splitset_labels = windowing.split_cases(
        input_df, train_frac, test_frac, svcases)
    
    ## plotting split data
    plot_choice = input('plot split data sets? (y/n) :')
    if plot_choice.lower() == 'y' or plot_choice.lower() == 'yes':
        windowing.plot_split_cases(input_df, fine_labels, splitset_labels, train_arr, val_arr, test_arr, 
                            features,svcases)
    else:
        pass
    
    ## Windowing hyperparameters
    steps_in, steps_out = 50, 50
    stride = 1
    
    #Windowed training data
    X_train, y_train, train_casebatch = windowing.window_data(train_arr, steps_in, stride, steps_out)
    #Windowed validation data
    X_val, y_val, val_casebatch = windowing.window_data(val_arr, steps_in, stride, steps_out)

    print(f"Windowed input training data shape: {X_train.shape}")
    print(f"Training windowed output shape: {y_train.shape}")
    print(f"Windowed input validation data shape: {X_val.shape}")
    print(f"Validation windowed output shape: {y_val.shape}")

    ######### LSTM MODEL TRAINING ##########

    # Define hyperparameters
    input_size = X_train.shape[-1]  # Number of features in the input tensor
    hidden_size = 128  # Number of hidden units in the LSTM cell, determines how many weights will be used in the hidden state calculations
    num_layers = 1 # Number of LSTM layers per unit
    output_size = y_train.shape[-1]  # Number of output features, same as input in this case
    pred_steps = steps_out # Number of future steps to predict
    batch_size = 30 # How many windows are being processed per pass through the LSTM
    learning_rate = 0.01
    num_epochs = 3000
    check_epochs = 100

    tf_ratio = 0.1
    dynamic_tf = False

    # customize loss function 
    penalty_weight = 10
    loss_fn = custom_loss(penalty_weight)
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
        
    ## Calling model class instance and training function
    model_choice = input('Select a LSTM model to train (DMS, S2S): ')

    if model_choice == 'DMS':
        # LSTM model instance
        model = LSTM_DMS(input_size, hidden_size, output_size, pred_steps,
                            l1_lambda=0.00, l2_lambda=0.00)
        
        optimizer = optim.Adam(model.parameters(), lr = learning_rate) # optimizer to estimate weights and biases (backpropagation)
            
        # Learning rate scheduler, set on min mode to decrease by factor when validation loss stops decreasing                                       
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        train_DMS(model=model, optimizer=optimizer, loss_fn=loss_fn, loader=loader, scheduler=scheduler, 
            num_epochs=num_epochs, check_epochs=check_epochs,
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            saveas='DMS_out')
        
    elif model_choice == 'S2S':
        # LSTM model instance
        model = LSTM_S2S(input_size, hidden_size, output_size, pred_steps)
        
        optimizer = optim.Adam(model.parameters(), lr = learning_rate) # optimizer to estimate weights and biases (backpropagation)
        
        # Learning rate scheduler, set on min mode to decrease by factor when validation loss stops decreasing                                       
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        train_S2S(model,optimizer, loss_fn, loader, scheduler, num_epochs, 
                  check_epochs,pred_steps,X_train,y_train, X_val, y_val,
                  training_prediction= 'mixed',tf_ratio=tf_ratio,
                  dynamic_tf=dynamic_tf,saveas='S2S_out')

    else:
        raise ValueError('Model selected is not configured/does not exist. Double check input.')

    ######## SAVING ALL RELEVANT DATA ########

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
        "steps_out": steps_out,
        "tf_ratio": tf_ratio,
        "dynamic_tf": False
    }

    with open(os.path.join(trainedmod_savepath,f'hyperparams_{model_choice}.txt'), "w") as file:

        for key, value in hyperparams.items():
            file.write(f"{key}: {value}\n")
    


if __name__ == "__main__":
    main()