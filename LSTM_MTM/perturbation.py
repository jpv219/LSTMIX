### LSTM perturbation and uncertainty analysis
### Authors: Fuyue Liang and Juan Pablo Valdes
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
from rollout_prediction import rollout
import numpy as np
from sklearn.metrics import r2_score
from contextlib import redirect_stdout
import io

# fig_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/figs/'
# trainedmod_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/trained_svmodels/'

fig_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/figs/'
trainedmod_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/trained_models/'

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


############################ Uncertainty Estimation #########################################################################

# Function to generate perturbed input sequence
def perturbation(num_pertb, pertb_scale, input_seq):
    '''
    Generate perturbed sequence where the perturbation is a random number from a uniform distribution
    
    : input pertb_scale: standard deviation for uniform distribution
    : return pertb_seqs: list of desired number of perturbed sequences
    '''
    pertb_seqs = []
    for _ in range(num_pertb):

        pertb = np.random.uniform(low=0, high=pertb_scale, size=input_seq.shape)
        pertb_seq = input_seq + pertb
        pertb_seqs.append(pertb_seq)
        
    return pertb_seqs

# Function to calculate uncertainty form perturbed input
def uncertainty_uni(pertb_preds):
    '''
    calcualte uncertainty of the trained model: perturbation scale based on uniform distribution
    
    : input pertb_preds: list of prediction from perturbed sequence; 
                         (num_pertb, 1, steps_in*num_forward, num_features)
    '''
    # convert the list of torch tensors into np.array
    pertb_preds = [t[0,:,:] for t in pertb_preds]
    pertb_preds = np.array([t.numpy() for t in pertb_preds])
    print('shape of perturbed prediction array: ', pertb_preds.shape)
    
    # mean predictions
    mean_preds = np.mean(pertb_preds, axis=0)
    
    # calculate percentiles for 95% prediction intervals
    lower = 2.5
    upper = 97.5
    
    lower_bound = np.percentile(pertb_preds, lower, axis=0)
    upper_bound = np.percentile(pertb_preds, upper, axis=0)
    
    return mean_preds, lower_bound, upper_bound

##################################### PLOTTING FUNC. ########################################
# Function to plot predictions from perturbed input emsemble
def plot_ensem_pred(model_name,features,scale,c_idx, case_label,
                    rollout_ref,true_data,rollouts_pertb):

    num_features = len(features)

    colors = sns.color_palette("vlag", len(rollouts_pertb))
    col = sns.color_palette("hsv",2)

    # Loop over features
    for f_idx in range(num_features):
        fig = plt.figure(figsize=(12,6))
        '''
        display r2_score between target and original prediction 
        plot target, predictions from original input, predictions from perturbed.
        '''

        plot_label = fine_labels.get(case_label,case_label)

        r2 = r2_score(true_data[:,c_idx,f_idx],rollout_ref[0,:,f_idx][:true_data.shape[0]])
        
        plt.plot(true_data[:,c_idx,f_idx],color = col[0], linewidth = 3, label=r'Target')

        plt.plot(rollout_ref[0,:,f_idx], '--',color = col[1], linewidth = 3, label=r'Ref rollout')

        for i in range(len(rollouts_pertb)):
            plt.plot(rollouts_pertb[i][0,:,f_idx], ':', color = colors[i % len(colors)])

        legend = plt.legend(ncol=2)
        legend.set_title(f'Perturbation scale:{scale:.2f}, $R^2$:{r2:.4f}, Case: {plot_label}')
        plt.title(f'Rollout pred with LSTM {model_name} for: {features[f_idx]}')
        plt.xlabel('Time steps')
        plt.ylabel(f'Scaled {features[f_idx]}')
        plt.xlim([0,110])
        plt.grid(color='k', linestyle=':', linewidth=0.1)
        plt.tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', length=5, width=1.5)

        fig.savefig(os.path.join(os.path.join(fig_savepath,'perturbations',model_name), 
                                 f'Perturbed_rollout_{model_name}_{features[f_idx]}.png'), dpi=150)
        plt.show()

# Function to plot uncertainty from perturbed rollout sequence
def plot_ensem_uncertainty(model_name,features,scale,c_idx, case_label,
                    rollout_ref,true_data,rollouts_pertb_mean,lower_bound,upper_bound):
    
    num_features = len(features)
    plot_label = fine_labels.get(case_label,case_label)
    col = sns.color_palette("hsv",2)
    
    # Loop over features
    for f_idx in range(num_features):
        '''
        plot target, original prediction, mean prediction, prediction interval
        '''
        fig = plt.figure(figsize=(12,6))
        plt.plot(true_data[:,c_idx,f_idx], color = col[0], lw=3, label=r'Target')
        plt.plot(rollout_ref[0,:,f_idx], color = col[1], linestyle='--', lw=3, label=r'Ref rollout')
        plt.plot(rollouts_pertb_mean[:, f_idx], 's',markersize=5,markerfacecolor='tab:red',
                     alpha=0.8,markeredgewidth=1.5, markeredgecolor='k',markevery=5,
                     lw=0.0, label=r'Mean perturbed rollout')

        # Prediction interval
        plt.fill_between(range(len(rollouts_pertb_mean)),lower_bound[:,f_idx], upper_bound[:,f_idx], 
                         color='orange',alpha=0.5, label=r'Prediction interval')
        
        legend = plt.legend(ncol=2)
        legend.set_title(f'Perturbation scale:{scale:.2f} for test case {plot_label}')
        plt.title(f'Uncertainty estimation of LSTM {model_name} for: {features[f_idx]}')
        plt.xlabel('Time steps')
        plt.ylabel(f'Scaled {features[f_idx]}')
        plt.xlim([0,110])
        plt.grid(color='k', linestyle=':', linewidth=0.1)
        plt.tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', length=5, width=1.5)

        fig.savefig(os.path.join(os.path.join(fig_savepath,'perturbations',model_name), f'Uncertainty_rollout_{model_name}_{features[f_idx]}.png'), dpi=150)
        plt.show()

########################################################################################
def main():
    
    features = ['Number of drops', 'Interfacial Area']

    ## Select LSTM model trained to use for predictions and plots
    model_choice = input('Select a LSTM model to use for perturbation analysis (DMS, S2S): ')

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
    
    ##### PERTURBATIONS #####
    num_pertb = 50 # number of perturbed sequences to generate
    pertb_scales = [0.2] # Perturbation scaling factor

    if model_choice == 'DMS':
        model = LSTM_DMS(hyperparams["input_size"], hyperparams["hidden_size"],
                         hyperparams["output_size"], hyperparams["pred_steps"],
                            l1_lambda=0.00, l2_lambda=0.00)
    elif model_choice == 'S2S':
        model = LSTM_S2S(hyperparams["input_size"], hyperparams["hidden_size"],
                         hyperparams["output_size"],hyperparams["pred_steps"])


    ## Load the last saved best model         
    model.load_state_dict(torch.load(os.path.join(trainedmod_savepath,f'{model_choice}_trained_model.pt')))

    ## Extract testing dataset
    test_arr = arrays[2]

    ## Extracting input steps from test data, keeping all cases and features in shape (in_steps,cases,features)
    c_idx = 1 # test case index for uncertainty estimation
    input_seq = test_arr[:hyperparams["steps_in"],c_idx:(c_idx+1),:]
    case_label = splitset_labels[2][c_idx]
    

    # Total steps to predict
    total_steps = test_arr.shape[0] - hyperparams["steps_in"]

    # Create a "null" object to redirect stdout
    null_output = io.StringIO()

    # Use the context manager to suppress prints
    with redirect_stdout(null_output):
        rollout_ref = rollout(model, input_seq, hyperparams["steps_out"], total_steps)

    ## ITERATION: perturbed prediction from perturbed input sequences
    pertb_perd_intvs = []

    for scale in pertb_scales: #to loop over different perturbation scales
        # generate perturbed input sequence
        pertb_input = perturbation(num_pertb, scale, input_seq) # List of perturbed input seq.
        
        rollouts_pertb = []

        # Rollingout predictions per perturbed input.
        for i in range(len(pertb_input)):
            with redirect_stdout(null_output):
                pertb_pred = rollout(model, pertb_input[i],hyperparams["steps_out"],total_steps) #Rollout prediction from perturbed input sequence [i]
            rollouts_pertb.append(pertb_pred) #saving iteration to list
        
        plot_ensem_pred(model_choice,features,scale,c_idx, case_label, rollout_ref,test_arr,rollouts_pertb)

        # evaluate uncertainty
        mean, lb, ub = uncertainty_uni(rollouts_pertb)
        
        pertb_perd_intv = np.absolute(ub-lb)
        pertb_perd_intvs.append(pertb_perd_intv) # perturbed output intervals

        plot_ensem_uncertainty(model_choice,features,scale,c_idx, case_label, rollout_ref,test_arr,mean,lb,ub)

if __name__ == "__main__":
    main()