### LSTM tools for training
### Author: Juan Pablo Valdes
### Code adapted from Fuyue Liang LSTM for stirred vessels
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import os
import torch
import torch.nn as nn

## Env. variables ##

# trainedmod_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/trained_models/'
# tuningmod_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/tuning/'

#trainedmod_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/trained_models/'

#trainedmod_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/trained_svmodels/'
trainedmod_savepath = '/content/fuyue_data/trained_svmodelsALLwC_0/'

##################################### CLASSES #################################################

class custom_loss(nn.Module):

    # constructor and super from the parent nn.Module class
    def __init__(self, penalty_weight):
        super().__init__()
        self.penalty_weight = penalty_weight
        
    def forward(self, prediction, target):
        # mse as base loss function
        mse_loss = nn.MSELoss()(prediction, target)#.to("cuda")
        
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
    def __init__(self, model_name, patience=2, verbose=False, delta=0.0001):
        """
        Args:
            patience (int): How long to wait after last improvement in the monitored metric.
                            Default: 2
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
        # Save the model's state_dict.
        torch.save(model.state_dict(), os.path.join(trainedmod_savepath,f'{self.name}_trained_model.pt'))
