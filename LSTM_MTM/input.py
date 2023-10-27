### LSTM pre-processing input data
### Author: Juan Pablo Valdes and Fuyue Liang
### First commit: Oct, 2023
### Department of Chemical Engineering, Imperial College London
#########################################################################################################################################################
#########################################################################################################################################################

import numpy as np
import pandas as pd
import Load_Clean_DF
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
import pickle
import random
import math

## Env. variables ##

fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
input_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/input_data/'

#fig_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/figs/'
#input_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX//LSTM_MTM/input_data/'

#fig_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/figs/'
#input_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/input_data/'

## Plot setup

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Computer Modern']})

SMALL_SIZE = 6
MEDIUM_SIZE = 12
BIGGER_SIZE = 18
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
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

##### CLASSES #####

class RawData_processing():

    def __init__(self,cases) -> None:
        self.cases = cases
        
    ## load raw data per case from csv files
    def import_rawdata(self,case):
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
    def sort_inputdata(self):

        # Initialize dicts to hold all data extracted from HR sims, before and after post-process
        pre_dict = {}
        post_dict = {}

        # Loop through all cases
        for case in self.cases:
            # Extract raw data
            df_Vol, Nd, IntA = self.import_rawdata(case)
            
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

class DSD_processing():

    def __init__(self,cases,num_bins) -> None:
        self.cases = cases
        self.num_bins = num_bins

    ## generate bins based on max/min volume sizes across all cases across all times
    def gen_bins(self,pre_dict,closed:str):
        
        #Scaling ref values for drop volumes
        l_cap = (0.035/(9.80665*(998-824)))**0.5
        v_cap = (4/3) * math.pi * (l_cap/2)**3

        max_list, min_list = [],[]

        #loop for all cases in pre_dict
        for case in self.cases:
            pre_dict[case]['Vol'] = pre_dict[case]['Vol'].apply(
                lambda x: np.log10(x/v_cap).astype('float32'))
            
            ## find max/min value in each volume array per time through lambda fun
            max_val = pre_dict[case]['Vol'].apply(lambda x: np.max(x)) #list of max value for each time
            min_val = pre_dict[case]['Vol'].apply(lambda x: np.min(x))

            # extracting max/min value for all times and appending per case
            max_list.append(max(max_val))
            min_list.append(min(min_val))
        
        ## bin range, max/min value for all cases
        max_size = math.ceil(max(max_list))
        min_size = math.floor(min(min_list))

        ## generate bins
        bins = pd.interval_range(start=min_size,
                periods=self.num_bins,end=max_size,closed=closed)
        
        #bin edges
        bin_edges = [interval.left for interval in bins]
        bin_edges.append(bins[-1].right)

        return pre_dict, bins, bin_edges
    
    def find_bin(self,value, bins):
        """ bins is a list of tuples, e.g., [(0,20), (20, 40), (40, 60)],
            binning returns the smallest index i of bins so that
            bin[i][0] <= value < bin[i][1]
        """
        for i in range(0, len(bins)):

            if value in bins[i]:
                return i
        return -1
    
    def sort_into_bins(self,pre_dict,post_dict):
        
        pre_dict, bins, bin_edges = self.gen_bins(pre_dict,closed='left')
        
        #initialize n_bin key in each case dict in pre_dict
        for case in self.cases:
            for i in range(1,self.num_bins + 1):
                pre_dict[case][f'b{i}'] = []
        

class postprocess():

    def __init__(self,cases,norm_columns,feature_map) -> None:
        self.cases = cases
        self.norm_columns = norm_columns
        self.feature_map = feature_map

    ## Normalize input data
    def scale_inputs(self,pre_dict,post_dict):
                
        for case in self.cases:
            norm_data_case = pre_dict[case]

            # Loop through each column to be normalized
            for column in self.norm_columns:

                mapped_column = self.feature_map.get(column)
                norm_data = norm_data_case[mapped_column].values.astype('float64')
                
                # Reshape the data to be compatible with the scaler
                norm_data = norm_data.reshape(-1, 1)
                
                # Create a MinMaxScaler and fit it to the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler = scaler.fit(norm_data)
                
                # Transform and store the normalized data in the post_data dict. Post dict only holds dictionary with input variables per case
                post_dict[case][mapped_column] = scaler.transform(norm_data).astype('float32')
        
        return post_dict

    ## shape input to a suitable format for LSTM MTM architecture

    def shape_inputdata(self,post_dict):

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

    def smoothing(self,data, method, window_size=None, poly_order=None, lowess_frac = None):
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

    #### PLOTS ####

    def plot_inputdata(self,fine_labels, data,dpi=150):
        ### looping over the number of features (Nd and IA)

        features = ['Number of drops', 'Interfacial Area']
        colors = sns.color_palette("husl", len(self.cases))

        # Create a single figure with multiple subplots
        fig, axes = plt.subplots(1,2, figsize=(12, 8), dpi=dpi, num=1)

        for i, ax in enumerate(axes):
            ax.set_title(f'{features[i]}')
            ax.set_xlabel('Time steps')
            ax.set_ylabel(f'Scaled {features[i]}')

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            for idx, case in enumerate(self.cases):

                label = fine_labels.get(case,case)
                ax.plot(data[:, idx, i], label=f'{label}', color=colors[idx % len(colors)])
                ax.tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', length=5, width=1.5)
                ax.grid(color='k', linestyle=':', linewidth=0.1)

        fig.suptitle(f'Input data: {features}', fontsize=18)
        axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=2,fontsize='xx-small')

        plt.tight_layout()
        plt.savefig(os.path.join(fig_savepath,'input_data'),dpi=dpi)
        plt.show()

    def plot_smoothdata(self,data, smoothed_data, fine_labels, method,dpi=150):
        
        fig, ax = plt.subplots(1,2, figsize=(12,8), dpi=dpi, num=2)
        colors = sns.color_palette("husl", len(self.cases))
        features = ['Number of drops', 'Interfacial Area']

        for feature in range(len(features)):
            for idx, _ in enumerate(self.cases):

                ax[0].plot(data[:,idx,feature],color=colors[idx % len(colors)])
                ax[0].set_title('Data before')
                ax[0].set_xlabel('Time steps')
                ax[0].set_ylabel('Scaled data')
                ax[0].tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', 
                                length=5, width=1.5)
                ax[0].grid(color='k', linestyle=':', linewidth=0.1)
                
                ax[1].plot(smoothed_data[:,idx,feature],color=colors[idx % len(colors)])
                ax[1].set_title('Data after')
                ax[1].set_xlabel('Time steps')
                ax[1].set_ylabel('Smoothed data')
                ax[1].tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', 
                                length=5, width=1.5)
                ax[1].grid(color='k', linestyle=':', linewidth=0.1)
        
        fig.suptitle(f'Smoothing method: {method}', fontsize=18)

        ax[0].legend(labels=[f'{fine_labels.get(case,case)}' for case in self.cases],
                    loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=2,fontsize='xx-small')

        for ax in ax:
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            
        plt.tight_layout()
        plt.savefig(os.path.join(fig_savepath,'smoothed_data'),dpi=dpi)
        plt.show()

## main ##

def main():

    Allcases = ['b03','b06','bi001','bi01','da01','da1','b06pm','b09pm','bi001pm',
    'bi1','bi01pm','3drop',
    'b09','da01pm','da001', 'coarsepm']

    cases = random.sample(Allcases,len(Allcases))

    feature_map = {'Number of drops': 'Nd',
                   'Interfacial Area': 'IA'
                   }
    norm_columns = ['Number of drops', 'Interfacial Area']

    ## raw data pre-processing, extracting and sorting from csv files
    rd_processor = RawData_processing(cases=cases)

    pre_dict,post_dict = rd_processor.sort_inputdata()

    DSD_choice = input('Include DSD in LSTM predictions? (y/n): ')

    # List of features to be normalized
    if DSD_choice.lower() == 'y':
        n_bins = 20
        
        ## building features to represent all DSD bins
        for i in range(1,n_bins+1):
            key = f'DSD_bin_{i}'
            norm_columns.append(key)
            feature_map[key] = f'b{i}' 

    # scaled input data 
    post_dict = scale_inputs(cases,norm_columns,feature_map)

    # re-shaped input data
    shaped_input = shape_inputdata(post_dict)

    #plotting
    plot_inputdata(cases,fine_labels,shaped_input)

    # smoothing data
    smoothed_data = smoothing(shaped_input,'savgol',window_size=5,poly_order=3)

    plot_smoothdata(shaped_input, smoothed_data,fine_labels, 'savgol', cases)

    ## saving input data 

    save_dict = {'smoothed_data' : smoothed_data,
                 'case_labels' : cases,
                 'features' : norm_columns}

    with open(os.path.join(input_savepath,'inputdata.pkl'),'wb') as file:
        pickle.dump(save_dict,file)

if __name__ == "__main__":
    main()