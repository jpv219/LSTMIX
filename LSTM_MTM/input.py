### LSTM pre-processing input data
### Author: Juan Pablo Valdes
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import numpy as np
import pandas as pd
import Load_Clean_DF
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
import pickle

## Env. variables ##

#fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
#input_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/input_data/'

fig_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/figs/'
input_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX//LSTM_MTM/input_data/'

### Methods ###

def import_rawdata(case):
    if case == '3drop' or case == 'coarsepm':
        # If true, extract only volume array
        df_Vol = Load_Clean_DF.extract_Vol(case)
    else:
        # If false, extract volume and concentration arrays
        df_Vol = Load_Clean_DF.extract_GVol(case)
    
    # Extract number of drops (Nd) and interfacial area (IntA)
    Nd = Load_Clean_DF.extract_Nd(case)
    IntA = Load_Clean_DF.extract_IA(case)

    return df_Vol, Nd, IntA

## group input data into case-wise dictionary

def sort_inputdata(cases):

    # Initialize dicts to hold all data extracted from HR sims, before and after post-process
    pre_dict = {}
    post_dict = {}

    # Loop through all cases
    for case in cases:
        # Extract raw data
        df_Vol, Nd, IntA = import_rawdata(case)
        
        time = Nd['Time']
        n_drops = Nd['Ndrops']
        IA = IntA['IA']
        DSD = df_Vol['Volume']
        
        # Determine if case needs surf. conc. or clean
        if case == '3drop' or case == 'coarsepm':
            G = []  # If true, set G as an empty list
        else:
            G = df_Vol['Gammatilde']  # If false, extract G data
        
        ## Dictionary holding a dictionary per case for all extracted data from paraview
        pre_dict[case] = {'Time': time, 'Nd': n_drops, 'IA': IA, 'Vol': DSD, 'G': G}
        
        # Initialize an empty post-process dict per case
        post_dict[case] = {}
    
    return pre_dict, post_dict

## Normalize input data

def scale_inputs(cases,norm_columns):
    
    pre_dict, post_dict = sort_inputdata(cases)

    for case in cases:
        norm_data_case = pre_dict[case]

        # Loop through each column to be normalized
        for column in norm_columns:
            norm_data = norm_data_case[column].values.astype('float64')
            
            # Reshape the data to be compatible with the scaler
            norm_data = norm_data.reshape(-1, 1)
            
            # Create a MinMaxScaler and fit it to the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler = scaler.fit(norm_data)
            
            # Transform and store the normalized data in the post_data dict. Post dict only holds dictionary with input variables per case
            post_dict[case][column] = scaler.transform(norm_data).astype('float32')
    
    return post_dict

## shape input to a suitable format for LSTM MTM architecture

def shape_inputdata(post_dict):

    array = []

    ### All cases must have the same number of data points for them to be used in LSTM
    min_length = min(len(data['IA']) for data in post_dict.values())

    # Iterate through each case, output from items() is a tuple containing case and corresponding features
    for case, features in post_dict.items():
        # Extract the 'Nd' and 'IA' data for the current case
        # Cases are truncated with the min length in order to be stacked as a nparray
        Nd_data = features['Nd'][:min_length]
        IA_data = features['IA'][:min_length]

        # Combine 'Nd_data' and 'IA_data' into a single numpy array per case, with each feature as a column. shape: rows, col =  timesteps ,features
        combined_data = np.column_stack((Nd_data, IA_data))
        
        # Append each combined dataset (2D nparray) per case as an element in a list.
        array.append(combined_data)

    # Re-shape array list as a 3D nparray, appending each case horizontally = stacking each case as a column, each column with 2 features 
    shaped_input = np.stack(array, axis=1)
    print('(time_step, num_case, num_feature)=', shaped_input.shape)

    return shaped_input

## smoothing function for shaped data

def smoothing(data, method, window_size=None, poly_order=None, lowess_frac = None):
    '''
    Input array : 2D array per feature, with shape (times, cases)
    Three methods to smooth:
    
    'moveavg': requires window_size
    'savgol': requires window_size, poly_order
    'lowess': requires lowess_frac
    '''

    ## rolling window averaging method
    if method == 'moveavg':
        if window_size is None:
            raise ValueError('Window size required')
        smoothed_data = pd.DataFrame(data).rolling(window_size, axis = 0).mean()
        smoothed_data.fillna(pd.DataFrame(data),inplace=True)
    ## SavGol filter based on fitting least-squares polynomial to a window of data points
    elif method == 'savgol':
        if window_size is None or poly_order is None:
            raise ValueError('Mising input arguments: Windowsize/polyorder')
        smoothed_data = np.apply_along_axis(
                    lambda col: savgol_filter(col, window_size, poly_order),
                    axis = 0, arr=data)
    ## Locally Weighted Scatterplot Smoothing, locally fitting linear regressions
    elif method == 'lowess':
        if lowess_frac is None:
            raise ValueError('Lowess fraction required')
        smoothed_data = np.apply_along_axis(
                    lambda col: lowess(col,np.arange(len(col)),frac = lowess_frac,return_sorted=False),
                    axis = 0, arr = data)
    else:
        raise ValueError('Unsupported smoothing method')
    
    return smoothed_data

## plots ##

def plot_inputdata(cases,data,dpi=150):
    ### looping over the number of features (Nd and IA)

    #Plot setup
    rc('text', usetex=True)
    custom_font = {'family': 'serif', 'serif': ['Computer Modern Roman']}
    rc('font', **custom_font)

    features = ['Number of drops', 'Interfacial Area']
    num_features = data.shape[-1]
    colors = sns.color_palette("husl", len(cases))

    # Create a single figure with multiple subplots
    fig, axes = plt.subplots(num_features, figsize=(10, 6 * num_features), dpi=dpi, num=1)
    plt.tight_layout()

    for i, ax in enumerate(axes):
        ax.set_title(f'{features[i]}')
        ax.set_xlabel('Time steps')
        ax.set_ylabel(f'Scaled {features[i]}')

        for case, idx in zip(cases, range(len(cases))):
            ax.plot(data[:, idx, i], label=f'{str(case)}', color=colors[idx % len(colors)])
            ax.legend()

    fig.suptitle(f'Input data: {features}', fontsize=18)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_savepath,'input_data'),dpi=dpi)
    plt.show()

def plot_smoothdata(data, smoothed_data, method, cases,dpi=150):
    
    fig, ax = plt.subplots(1,2, figsize=(10,6), dpi=dpi, num=2)
    colors = sns.color_palette("husl", len(cases))

    for case, idx in zip(cases, range(len(cases))):
        ax[0].plot(data[:,idx], label = f'{str(case)}',color=colors[idx % len(colors)])
        ax[0].set_title('Data before')
        ax[0].set_xlabel('Time steps')
        ax[0].set_ylabel('Scaled data')
        
        ax[1].plot(smoothed_data[:,idx], label = f'{str(case)}',color=colors[idx % len(colors)])
        ax[1].set_title('Data after')
        ax[1].set_xlabel('Time steps')
        ax[1].set_ylabel('Smoothed data')
    
    fig.suptitle(f'Smoothing method: {method}', fontsize=18)

    ax[0].legend()
    ax[1].legend()
    plt.savefig(os.path.join(fig_savepath,'smoothed_data'),dpi=dpi)
    plt.tight_layout()
    plt.show()

## main ##

def main():

    Allcases = ['b03','b06','bi001','bi01','da01','da1','b06pm','b09pm','bi001pm',
    'bi1','bi01pm','3drop',
    'b09','da01pm','da001', 'coarsepm']

    # List of columns to be normalized
    norm_columns = ['Nd', 'IA']

    # scaled input data 
    post_dict = scale_inputs(Allcases,norm_columns)

    # re-shaped input data
    shaped_input = shape_inputdata(post_dict)

    #plotting
    plot_inputdata(Allcases,shaped_input)

    # smoothing data
    smoothed_data = smoothing(shaped_input,'savgol',window_size=5,poly_order=3)

    plot_smoothdata(shaped_input, smoothed_data, 'savgol', Allcases)

    ## saving input data 

    with open(os.path.join(input_savepath,'inputdata.pkl'),'wb') as file:
        pickle.dump(smoothed_data,file)

if __name__ == "__main__":
    main()