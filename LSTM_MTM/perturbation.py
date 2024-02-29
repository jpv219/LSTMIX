### LSTM perturbation and uncertainty analysis
### Authors: Fuyue Liang and Juan Pablo Valdes
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import os
import pickle
import torch
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from modeltrain_LSTM import LSTM_FC, LSTM_ED, GRU_FC, GRU_ED
from rollout_prediction import Rollout,PathConfig
import numpy as np
from sklearn.metrics import r2_score
from contextlib import redirect_stdout
import io
import configparser

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
    'bi1':r'$Bi=1$','bi01pm':r'$Bi_{pm}=0.1$,','3d':r'3-Drop',
    'b09':r'$\beta=0.9$','da01pm':r'$Da_{pm}=0.1$, ','da001':r'$Da=0.01$', 'PM':r'Coarse Pre-Mix', 'FPM' : r'Fine Pre-Mix',
    'alt1': r'Alt1', 'alt2': r'Alt2', 'alt3': r'Alt3', 'alt4': r'Alt4', 'alt1_b09': r'$\beta_{alt1 pm}=0.9$', 'alt4_b09':  r'$\beta_{alt4 pm}=0.9$',
    'alt4_f': r'Alt4 Fine PM', 'b03a': r'$\beta_{alt4}=0.3$', 'b06a': r'$\beta_{alt4}=0.6$', 'b09a': r'$\beta_{alt4}=0.9$',
    'bi1a': r'$Bi_{alt4}=1', 'bi01a': r'$Bi_{alt4}=01', 'bi001a': r'$Bi_{alt4}=0.01'
}

feature_labels = {
    'ND': r'${ND}$', 'IA':r'${IA}$', 'Range 3': r'$B_3$','Range 5': r'$B_5$', 'Range 6': r'$B_6$', 'Range 8': r'$B_8$'
}

class Perturbation(Rollout):
    # Constructor
    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

############################ Uncertainty Estimation #########################################################################

    # Function to generate perturbed input sequence
    def perturbation(self,num_pertb, pertb_scale, input_seq):
        '''
        Generate perturbed sequence where the perturbation is a random number from a uniform distribution
        
        : input pertb_scale: standard deviation for uniform distribution
        : return pertb_seqs: list of desired number of perturbed sequences
        '''
        pertb_seqs = []
        np.random.seed(38)
        for _ in range(num_pertb):
            pertb_seqs_mem = []
            for t in range(0,len(input_seq)):
                unpertb_value = input_seq[t,:,:]
                unpertb_seq = np.full((num_pertb,unpertb_value.shape[-1]),unpertb_value)
                pertb = np.random.normal(loc=0, scale=pertb_scale, size=unpertb_seq.shape)
                pertb_seq = unpertb_seq + pertb
                pertb_seqs_mem.append(pertb_seq)
            pertb_seqs.append(pertb_seqs_mem)
            
        return pertb_seqs

    # Function to calculate uncertainty from perturbed input
    def uncertainty_norm(self,pertb_preds):
        '''
        calculate uncertainty of the trained model: perturbation scale based on normal distribution
        
        : input pertb_preds: list of prediction from perturbed sequence; 
                            (num_pertb, 1, steps_in*num_forward, num_features)
        : return arrays of mean, std, lower bound and upper bound for prediction intervals; 
                            (steps_in*num_forward, num_features)
        '''
        # convert the list of torch tensors into np.array for mean and std calculation
        pertb_preds = [t[0,:,:] for t in pertb_preds]
        pertb_preds = np.array([t.numpy() for t in pertb_preds])
        print('shape of perturbed prediction array: ', pertb_preds.shape)
        
        # Calculate mean and standard deviation across the samples for each time step
        mean_preds = np.mean(pertb_preds, axis=0)
        std_preds = np.std(pertb_preds, axis=0)

        # Optionally, can calculate prediction intervals
        confi_int = 2 # For a 95% confidence interval corresponding to standard normal distribution
        lower_bound = mean_preds - confi_int * std_preds
        upper_bound = mean_preds + confi_int * std_preds
        
    #     print('mean and std shape: ', mean_preds.shape, std_preds.shape)
        return mean_preds, lower_bound, upper_bound

    # Function to calculate uncertainty form perturbed input
    def uncertainty_uni(self,pertb_preds):
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
    def plot_combo_uncertainty(self,model_name,c_idx,features,
                    rollout_ref,true_data,rollouts_pertb,steps_in,rollouts_pertb_mean,lower_bound,upper_bound):
    
        #Path constructor
        path = PathConfig()

        num_features = 2 #Nd and IA
        fig,ax = plt.subplots(figsize=(12,7))
        colors = sns.color_palette('Paired',num_features*3)
        ax1 = ax.twinx()
        '''
        plot target, original prediction, mean prediction, prediction interval
        '''
        y_ax = feature_labels.get(features[0],features[0])
        colorline_ax = colors[1]
        colorband_ax = colors[0]

        for i in range(len(rollouts_pertb)):
            ax.plot(rollouts_pertb[i][0,:steps_in,0], linewidth=1.5, linestyle=':')

        ax.plot(true_data[:,c_idx,0], color=colorline_ax, lw=3, label=r'Target')
        ax.plot(rollout_ref[0,:,0], 's',markersize=5,markerfacecolor=colorline_ax,
                alpha=0.8,markeredgewidth=1.5, markeredgecolor='k',markevery=5,lw=0.0,
                label=r'Unperturbed prediction')
        # Prediction interval
        ax.fill_between(range(steps_in,len(rollouts_pertb_mean)),lower_bound[steps_in:,0], upper_bound[steps_in:,0],
                            color=colorband_ax,alpha=0.7, label=r'Prediction interval region')
        
        ax.set_ylabel(f'{y_ax}',fontsize=40,color=colorline_ax)
        ax.tick_params(bottom=True, top=True, left=True, right=False,axis='y',direction='in', length=5, width=1.5,labelcolor=colorline_ax,labelsize=25)
        ##############################################################
        y_ax1 = feature_labels.get(features[1],features[1])
        colorline_ax1 = colors[5]
        colorband_ax1 = colors[4]

        for i in range(len(rollouts_pertb)):
            ax1.plot(rollouts_pertb[i][0,:steps_in,1], linewidth=1.5,linestyle=':')

        ax1.plot(true_data[:,c_idx,1], color=colorline_ax1, lw=3, label=r'Target')
        ax1.plot(rollout_ref[0,:,1], 's',markersize=5,markerfacecolor=colorline_ax1,
                alpha=0.8,markeredgewidth=1.5, markeredgecolor='k',markevery=5,lw=0.0,
                label=r'Unperturbed prediction')
        # Prediction interval
        ax1.fill_between(range(steps_in,len(rollouts_pertb_mean)),lower_bound[steps_in:,1], upper_bound[steps_in:,1],zorder=-10,
                            color=colorband_ax1,alpha=0.7, label=r'95$\%$ Prediction interval')
        
        ax1.set_ylabel(f'{y_ax1}',fontsize=40,labelpad=20,color=colorline_ax1)
        ax1.tick_params(bottom=True, top=True, left=False, right=True,axis='y',direction='in', length=5, width=1.5,labelcolor=colorline_ax1,labelsize=25)
        
        h, l = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax1.get_legend_handles_labels())]
        plt.legend(title=f'{model_name}: {self.mixer}',title_fontsize=20,
                handles=zip(h[:3],h[3:]), labels=l[:3], # ::2 every two elements
                handler_map = {tuple: mpl.legend_handler.HandlerTuple(None)},
                fontsize=18,loc='lower right',
                edgecolor='black', frameon=True)
        
        ax.set_xlabel('Time steps',fontsize=40)
        ax.tick_params(bottom=True, top=True, left=True, right=True,axis='x',direction='in', length=5, width=1.5,labelsize=25)
        
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.join(path.fig_savepath,'perturbations',model_name), f'Uncertainty_{model_name}_Combo.png'), dpi=150)
        plt.show()

    def plot_residuals(self,model_name,features,residuals_origin,steps_in,total_steps,pertb_perd_intvs,count):
        
        path = PathConfig()
        
        fig, ax = plt.subplots(1,len(features),figsize=(int(len(features)*8),8))
        fig.subplots_adjust(wspace=0.3)
        cmap = sns.color_palette('rocket', 6)
        for f_idx in range(len(features)):
            y_label = feature_labels.get(features[f_idx],features[f_idx])

            ax[f_idx].plot(residuals_origin[steps_in:,f_idx],color=cmap[0],lw=3,label=f'Absolute residual')
            ax[f_idx].set_ylabel(f'{y_label}', fontsize=30)
            
            ax[f_idx].plot(pertb_perd_intvs[count][steps_in:,f_idx], '--', color=cmap[5],lw=3,label=f'Prediction interval')
            # corrcoef = np.corrcoef(residuals_origin[steps_in:,f_idx], pertb_perd_intvs[count][steps_in:steps_in+total_steps,f_idx])[0,1]
            pearr=stats.pearsonr(residuals_origin[steps_in:,f_idx], pertb_perd_intvs[count][steps_in:steps_in+total_steps,f_idx])
            spearr=stats.spearmanr(residuals_origin[steps_in:,f_idx], pertb_perd_intvs[count][steps_in:steps_in+total_steps,f_idx])
            ax[f_idx].set_title(f'Pearson\'s coefficient = {pearr.statistic:.2f}', fontsize=30)
            # ax[f_idx].set_title(f'Spearman\'s coefficient = {spearr.statistic:.2f}', fontsize=30)
            ax[f_idx].tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', length=5, width=1.5, labelsize=20)
            print(f'PCC:{y_label}:{spearr}')

        ax[0].legend(title=f'{model_name}: {self.mixer}', title_fontsize=20, fontsize=18,edgecolor='black', frameon=True)

        fig.supxlabel('Time step',fontsize=30)
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.join(path.fig_savepath,'perturbations',model_name), f'Residuals_{model_name}.png'), dpi=150)
        plt.show()
########################################################################################
def main():
    
    #Path constructor
    path = PathConfig()
    
    features = ['Number of drops', 'Interfacial Area']

    ## Selection of Neural net architecture and RNN unit type
    mixer_choice = input('UQ for which mixer? (sv,sm): ')
    titles = {'sm': 'Static Mixer', 'sv': 'Stirred Mixer'}
    mixer_title = titles.get(mixer_choice,'Input error')

    net_choice = input('Select a network to use for predictions (GRU/LSTM): ')

    arch_choice = input('Select the specific network architecture (FC/ED): ')

    model_choice = net_choice + '_' + arch_choice

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
    with open(os.path.join(path.trainedmod_savepath,f'hyperparams_{model_choice}.txt'), "r") as file:
        for line in file:
            key, value = line.strip().split(": ")  # Split each line into key and value
            hyperparams[key] = eval(value)

    ##Renaming recurrent hyperparameters for legibility
    input_size = hyperparams["input_size"]
    hidden_size = hyperparams["hidden_size"]
    output_size = hyperparams["output_size"]
    pred_steps = hyperparams["pred_steps"]
    steps_in = hyperparams["steps_in"]
    steps_out = hyperparams["steps_out"]
    l1 = hyperparams["l1"]
    l2 = hyperparams["l2"]

    ## Loading numpyarrays for all split datasets and labels, as well as windowed training and val tensors with casebatch lengths
    for setlbl in set_labels:
        npfile = os.path.join(path.trainedmod_savepath,f'data_sets_{model_choice}', f'{setlbl}_pkg.pkl')
        ptfile = os.path.join(path.trainedmod_savepath,f'data_sets_{model_choice}', f'X_{setlbl}.pt')

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

    # Rollout instance        
    rollout = Rollout(input_size,output_size,steps_in,steps_out,mixer_title)
    pertb = Perturbation(input_size,output_size,steps_in,steps_out,mixer_title)

    num_pertb = 200 # number of perturbed sequences to generate
    pertb_scales = [0.04] # Perturbation scaling factor

    if model_choice == 'LSTM_FC':
        model = LSTM_FC(input_size, hidden_size,output_size, pred_steps,
                            l1_lambda=l1, l2_lambda=l2)
    elif model_choice == 'LSTM_ED':
        model = LSTM_ED(input_size, hidden_size,output_size, pred_steps,
                            l1_lambda=l1, l2_lambda=l2)
    elif model_choice == 'GRU_FC':
        model = GRU_FC(input_size, hidden_size,output_size, pred_steps,
                            l1_lambda=l1, l2_lambda=l2)
    elif model_choice == 'GRU_ED':
        model = GRU_ED(input_size, hidden_size,output_size, pred_steps,
                            l1_lambda=l1, l2_lambda=l2)


    ## Load the last saved best model         
    model.load_state_dict(torch.load(os.path.join(path.trainedmod_savepath,f'{model_choice}_trained_model.pt')))

    ## Extract testing dataset
    test_arr = arrays[2]

    ## Extracting input steps from test data, keeping all cases and features in shape (in_steps,cases,features)
    c_idx = 1 # test case index for uncertainty estimation
    input_seq = test_arr[:steps_in,c_idx:(c_idx+1),:]
    case_label = splitset_labels[2][c_idx]
    

    # Total steps to predict
    total_steps = test_arr.shape[0] - steps_in

    # Create a "null" object to redirect stdout
    null_output = io.StringIO()

    # Use the context manager to suppress prints
    with redirect_stdout(null_output):
        rollout_ref = rollout.rollout(model, input_seq, total_steps)
        ## calculate the residuals
        residuals_origin = np.absolute(np.array(test_arr[:,c_idx,:]) - np.array(rollout_ref[0,:,:][:test_arr.shape[0]]))


    ## ITERATION: perturbed prediction from perturbed input sequences
    pertb_perd_intvs = []

    noise_choice = input('Select a distribution from which the perturbed noises are drawn (norm/uni):')

    for count, scale in enumerate(pertb_scales): #to loop over different perturbation scales
        # generate perturbed input sequence
        pertb_input = pertb.perturbation(num_pertb, scale, input_seq) # List of perturbed input seq.
        
        rollouts_pertb = []

        # Rollingout predictions per perturbed input.
        for i in range(len(pertb_input)):
            with redirect_stdout(null_output):
                pertb_pred = rollout.rollout(model, pertb_input[i],total_steps) #Rollout prediction from perturbed input sequence [i]
            rollouts_pertb.append(pertb_pred) #saving iteration to list

        # evaluate uncertainty
        if noise_choice == 'norm':
            mean, lb, ub = pertb.uncertainty_norm(rollouts_pertb)
        elif noise_choice == 'uni':
            mean, lb, ub = pertb.uncertainty_uni(rollouts_pertb)
        
        pertb_perd_intv = np.absolute(ub-lb)
        pertb_perd_intvs.append(pertb_perd_intv) # perturbed output intervals

        pertb.plot_combo_uncertainty(model_choice,c_idx,features[:2],rollout_ref,test_arr,rollouts_pertb, hyperparams["steps_in"], mean,lb,ub)
        pertb.plot_residuals(model_choice,features[:2],residuals_origin,hyperparams["steps_in"],total_steps,pertb_perd_intvs,count)

if __name__ == "__main__":
    main()