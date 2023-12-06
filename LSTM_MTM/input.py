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
import copy

## Env. variables ##

#fig_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/figs/'
#input_savepath = '/Users/mfgmember/Documents/Juan_Static_Mixer/ML/LSTM_SMX/LSTM_MTM/input_data/'

#fig_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX/LSTM_MTM/figs/'
#input_savepath = '/Users/juanpablovaldes/Documents/PhDImperialCollege/LSTM/LSTM_SMX//LSTM_MTM/input_data/'

fig_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/figs/'
input_savepath = '/home/jpv219/Documents/ML/LSTM_SMX/LSTM_MTM/input_data/'
raw_datapath = '/home/jpv219/Documents/ML/LSTM_SMX/RawData'

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
    'bi1':r'$Bi=1$','bi01pm':r'$Bi_{pm}=0.1$,','3d':r'3-Drop',
    'b09':r'$\beta=0.9$','da01pm':r'$Da_{pm}=0.1$, ','da001':r'$Da=0.01$', 'PM':r'Coarse Pre-Mix', 'FPM' : r'Fine Pre-Mix',
    'alt1': r'Alt1', 'alt2': r'Alt2', 'alt3': r'Alt3', 'alt4': r'Alt4', 'alt1_b09': r'$\beta_{alt1 pm}=0.9$', 'alt4_b09':  r'$\beta_{alt4 pm}=0.9$',
    'alt4_f': r'Alt4 Fine PM', 'b03a': r'$\beta_{alt4}=0.3$', 'b06a': r'$\beta_{alt4}=0.6$', 'b09a': r'$\beta_{alt4}=0.9$',
    'bi1a': r'$Bi_{alt4}=1$', 'bi01a': r'$Bi_{alt4}=0.1$', 'bi001a': r'$Bi_{alt4}=0.01$'
}

##### CLASSES #####

class RawData_processing():

    def __init__(self,cases) -> None:
        self.cases = cases
        
    ## load raw data per case from csv files
    def import_rawdata(self,case):

        file_name_gvol = os.path.join(raw_datapath,f"{case}_GVol.csv")
        file_name_vol = os.path.join(raw_datapath,f"{case}_Vol.csv")

    # Check if the files exist
        if os.path.isfile(file_name_gvol):
            # If true, extract volume and concentration arrays
            df_Vol = Load_Clean_DF.extract_GVol(case)

        elif os.path.isfile(file_name_vol):
            # If false, extract only volume array
            df_Vol = Load_Clean_DF.extract_Vol(case)
        else:
            # Handle the case where neither file exists
            raise FileNotFoundError(f"Neither {file_name_gvol} nor {file_name_vol} found.")
        
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

            file_name_gvol = os.path.join(raw_datapath,f"{case}_GVol.csv")
            file_name_vol = os.path.join(raw_datapath,f"{case}_Vol.csv")
            
            # Determine if case needs surf. conc. or clean
            if os.path.isfile(file_name_vol):
                G = []  # If true, set G as an empty list
            elif os.path.isfile(file_name_gvol):
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
   
    def sort_into_bins(self,pre_dict):
        
        pre_dict, bins, bin_edges = self.gen_bins(pre_dict,closed='left')
        
        # Loop through all case dicts
        for case in self.cases:
            # Initialize a dictionary for bin counts per timestep (size based on total times from len(predict))
            bin_counts = {f'b{i}': [0] * len(pre_dict[case]['Vol']) for i in range(self.num_bins)}
            
            # Loop through volume arrays in each time j in each case
            for j, d_vol_array in enumerate(pre_dict[case]['Vol']):

                # Classify each drop volume into a given bin for timestep j
                for drop in d_vol_array:
                    bin_index = self.find_bin(drop, bins)
                    bin_counts[f'b{bin_index}'][j] += 1 # store the bin count at each time j
            
            # Assign the counts to pre_dict[case]
            for key, value in bin_counts.items():
                pre_dict[case][key] = value

        return pre_dict, bin_edges

    def filter_bins(self,bin_edges, pre_dict, leftmost, rightmost):

        ## Extract the bin_width
        bin_width = np.diff(bin_edges[leftmost:rightmost+1])[-1]

        for case in self.cases:
            bins_kept = []
            bins_to_delete = []

            for key in pre_dict[case].keys():
                ## targeting only keys corresponding to bins
                if key.startswith('b'):
                    bin_number = int(key[1:])  # Extract the number after 'b'

                    ## deleting bins based on left/rightmost filters
                    if bin_number < leftmost or bin_number > rightmost:
                        bins_to_delete.append(key)
                    ## Keep the rest of the bins
                    else:
                        bins_kept.append(key)

            ## filter out bins
            for key in bins_to_delete:
                del pre_dict[case][key]
        
        return pre_dict,bin_width,bins_to_delete, bins_kept

    def density_func_est(self,f_pre_dict, bin_width, bins_kept):
        '''
        Transform the drop counts into probabilities
        Input: pre_dict containing dictionaries per case
        leftmost: smallest bin number
        rightmost: largest bin number
        output: pre_dict with bins replaced for density fun estimates
        '''

        ## Carrying out density fun estimation per case for all bins
        for case in self.cases:
            prob_dens = []
            norm_dens = []

            i=0

            for key in f_pre_dict[case].keys():

                ## targeting only keys corresponding to bins
                if key.startswith('b'):
                    ## calculate prob density estimation per bin: prob_dens is a list containing bin arrays.
                    prob_dens.append(np.where(f_pre_dict[case]['Nd']>0, f_pre_dict[case][key]/f_pre_dict[case]['Nd']*bin_width,0))

            ## Calculate cumulative density values for all bins per time: element-wise addition of all bins per time
            cum_dens = np.zeros_like(prob_dens[0])
            for arr in prob_dens:
                cum_dens += arr

            ## Go through each prob density bin array and normalize it
            for j in range(len(prob_dens)):

                arr = np.where(cum_dens>0, prob_dens[j] / cum_dens, 0)
                norm_dens.append(arr)

            for i, bin in enumerate(bins_kept):
                ## re-write bin drop counts with norm density values
                f_pre_dict[case][bin] = norm_dens[i]

        return f_pre_dict

class Post_processing():

    def __init__(self,cases,norm_columns,feature_map,DSD_choice:str,DSD_columns) -> None:
        self.cases = cases
        self.norm_columns = norm_columns
        self.DSD_columns = DSD_columns
        self.feature_map = feature_map
        self.DSD_choice = DSD_choice

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
            
            ## if DSD is being processed, transfer bins into post_dict
            if self.DSD_choice.lower() == 'y':

                for col in self.DSD_columns:
                    mapped_col = self.feature_map.get(col)
                    post_dict[case][mapped_col] = norm_data_case[mapped_col].astype('float32')

                
        return post_dict

    ## shape input to a suitable format for LSTM MTM architecture

    def shape_inputdata(self,post_dict):

        array = []
        ## Extracting all keys to be processed from normalised features and filtered bins.
        keys = [self.feature_map.get(feature) for feature in self.norm_columns+self.DSD_columns] 

        ### All cases must have the same number of data points for them to be used in LSTM
        min_length = min(len(data['IA']) for data in post_dict.values())

        # Iterate through each case
        for case in self.cases:
            case_data = []
            #Iterate per feature per case
            for key in keys:
                # Cases are truncated with the min length in order to be stacked as a nparray
                data = post_dict[case][key][:min_length]
                case_data.append(data)

            # Combine all features per case into a single numpy array, with each feature as a column. shape: rows, col =  timesteps ,features
            combined_data = np.column_stack(case_data)
            
            # Append each combined dataset (2D nparray) per case as an element in a list.
            array.append(combined_data)

        # Re-shape array list as a 3D nparray, appending each case horizontally = stacking each case as a column, each column with n features 
        shaped_input = np.stack(array, axis=1)
        print('(time_step, num_case, num_feature)=', shaped_input.shape)

        return shaped_input

    ## smoothing function for shaped data

    def smoothing(self,in_data, method, window_size=None, poly_order=None, lowess_frac = None):
        '''
        Input array : array with shape times,cases,features, smoothing only features 1,2 (ND,IA)
        Three methods to smooth:
        
        'moveavg': requires window_size
        'savgol': requires window_size, poly_order
        'lowess': requires lowess_frac
        '''
        # Create a copy of the input data to avoid modifying the original array
        data = np.copy(in_data)
        # Extract the features to be smoothed
        smoothed_features = data[:, :, :2]
        ## rolling window averaging method
        if method == 'moveavg':
            if window_size is None:
                raise ValueError('Window size required')
            smoothed_data = pd.DataFrame(smoothed_features).rolling(window_size, axis = 0).mean()
            smoothed_data.fillna(pd.DataFrame(smoothed_features),inplace=True)
        ## SavGol filter based on fitting least-squares polynomial to a window of data points
        elif method == 'savgol':
            if window_size is None or poly_order is None:
                raise ValueError('Mising input arguments: Windowsize/polyorder')
            smoothed_data = np.apply_along_axis(
                        lambda col: savgol_filter(col, window_size, poly_order),
                        axis = 0, arr=smoothed_features)
        ## Locally Weighted Scatterplot Smoothing, locally fitting linear regressions
        elif method == 'lowess':
            if lowess_frac is None:
                raise ValueError('Lowess fraction required')
            smoothed_data = np.apply_along_axis(
                        lambda col: lowess(col,np.arange(len(col)),frac = lowess_frac,return_sorted=False),
                        axis = 0, arr = smoothed_features)
        else:
            raise ValueError('Unsupported smoothing method')
        
        ## Locate range/interval of the smoothed columns
        smoothed_columns = np.arange(smoothed_data.shape[-1])

        # Reassign the smoothed columns to the original data
        data[:, :, smoothed_columns] = smoothed_data
        
        return data

    #### PLOTS ####

    def plot_inputdata(self,fine_labels, data,dpi=150):
        ### looping over the number of features (Nd and IA)

        colors = sns.color_palette("husl", len(self.cases))

        # Create a single figure with multiple subplots
        fig, axes = plt.subplots(1,2, figsize=(12, 8), dpi=dpi, num=1)

        for i, ax in enumerate(axes):
            ax.set_title(f'{self.norm_columns[i]}')
            ax.set_xlabel('Time steps')
            ax.set_ylabel(f'Scaled {self.norm_columns[i]}')

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            for idx, case in enumerate(self.cases):

                label = fine_labels.get(case,case)
                ax.plot(data[:, idx, i], label=f'{label}', color=colors[idx % len(colors)])
                ax.tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', length=5, width=1.5)
                ax.grid(color='k', linestyle=':', linewidth=0.1)

        fig.suptitle(f'Input data: {self.norm_columns}', fontsize=18)
        axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=2,fontsize='xx-small')

        plt.tight_layout()
        plt.savefig(os.path.join(fig_savepath,'input_data','input_Nd_IA'),dpi=dpi)
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
        plt.savefig(os.path.join(fig_savepath,'input_data','smoothed_data'),dpi=dpi)
        plt.show()

    def plot_DSD(self,data,bin_edges,fine_labels,dpi=150):
        
        t_indices = [65,75,85,95]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        for idx, t_idx in enumerate(t_indices):

            row = idx // 2
            col = idx % 2

            for j, case in enumerate(self.cases):
                label = fine_labels.get(case)

                ax = axes[row, col]

                ax.hist(bin_edges, bins=len(bin_edges), fill=False,weights=data[t_idx, j, 2:].tolist(), label=f'{label}')
                ax.set_ylabel('Drop count density function')
                ax.set_xlabel(r'$Log_{10}(V/V_{cap})$')
                ax.legend()
                ax.set_title(f'DSD at time {t_idx*0.005} s')

        plt.tight_layout()
        plt.savefig(os.path.join(fig_savepath,'input_data','DSD_data'),dpi=dpi)
        plt.show()

## SETUP DSD METHOD ##

def setup_DSD(n_bins,cases,feature_map,DSD_columns,pre_dict):
    
    ## building features to represent all DSD bins
    for i in range(0,n_bins):
        key = f'b{i}'
        DSD_columns.append(key)
        feature_map[key] = key
    
    ## DSD processing class
    DSD_processor = DSD_processing(cases=cases, num_bins=n_bins)
    leftmost=1
    rightmost=10

    ## Pre_dict with bins and drop counts assigned: dc - dropcounts
    pre_dict_dc, bin_edges = DSD_processor.sort_into_bins(pre_dict)

    ## Filter bins based on left/right most limits
    pre_dict_dc, bin_width, bins_to_delete, bins_kept = DSD_processor.filter_bins(bin_edges,pre_dict_dc,leftmost,rightmost)

    ## Save a copy of pre_dict with only drop counts just for plotting
    dc_copy = copy.deepcopy(pre_dict_dc)

    ## Convert bin drop counts into probability density function values
    pre_dict = DSD_processor.density_func_est(pre_dict_dc,bin_width, bins_kept)

    ## delete bins filtered out from original bin list
    DSD_columns = [item for item in DSD_columns if item not in bins_to_delete]
    bin_edges = bin_edges[leftmost:rightmost+1]

    return dc_copy, DSD_columns, bin_edges, feature_map

## main ##

def main():

    Allcases = ['bi001', 'bi01', 'b09', 'b06pm', 'b03', 'da01pm', 'da01', 'bi01pm', '3d', 'alt1', 'alt4_b09','b03a','b09a','bi01a','bi1a',
        'PM', 'bi001pm', 'bi1', 'alt3','alt1_b09','alt4_f','b06a',
        'b06', 'b09pm', 'da1', 'da001','alt2','bi001a','FPM']

    # Randomizing cases for different train-test set splitting
    cases = random.sample(Allcases,len(Allcases))

    # List of features to be normalized (without DSD)
    feature_map = {'Number of drops': 'Nd',
                   'Interfacial Area': 'IA'
                   }
    norm_columns = ['Number of drops', 'Interfacial Area']

    DSD_columns = []

    ######## RAW DATA PROCESSING ######

   ## raw data pre-processing, extracting and sorting from csv files
    rd_processor = RawData_processing(cases=Allcases)

    ## post dict empty with case slots built in
    pre_dict,post_dict = rd_processor.sort_inputdata()

    ######## DSD DATA PROCESSING ######

    DSD_choice = input('Include DSD in LSTM predictions? (y/n): ')

    ## Including DSD data for LSTM prediction: pre-process and data preparation
    if DSD_choice.lower() == 'y':

        n_bins = 12
        dc_copy, DSD_columns, bin_edges, feature_map = setup_DSD(n_bins,Allcases,feature_map,DSD_columns,pre_dict)

    ######## POST-PROCESSING ######

    post_processor = Post_processing(cases=Allcases,
                    norm_columns=norm_columns,feature_map=feature_map,
                    DSD_choice=DSD_choice,DSD_columns=DSD_columns)

    # scaled input data, in case of DSD: include processed bins into post_dict
    post_dict = post_processor.scale_inputs(pre_dict,post_dict)

    # re-shaped input data
    shaped_input = post_processor.shape_inputdata(post_dict)

    ##### DSD PLOTTING #####

    if DSD_choice.lower() == 'y':

        ## reshape pre_dict with bins into numpy array for handling and plotting
        shaped_data_dc = post_processor.shape_inputdata(dc_copy)

        ## plot drop count histogram
        post_processor.plot_DSD(shaped_data_dc,bin_edges,fine_labels)
        ##plot density function histogram
        post_processor.plot_DSD(shaped_input,bin_edges,fine_labels)

    #### DATA SMOOTHING AND SAVING

    #plotting
    post_processor.plot_inputdata(fine_labels,shaped_input)

    # smoothing data
    smoothed_data = post_processor.smoothing(shaped_input,'savgol',window_size=5,poly_order=3)

    post_processor.plot_smoothdata(shaped_input, smoothed_data,fine_labels, 'savgol')

    ## saving input data 

    save_dict = {'smoothed_data' : smoothed_data,
                 'case_labels' : Allcases,
                 'features' : norm_columns+DSD_columns}
    if DSD_choice.lower() == 'y':
        save_dict['bin_edges'] = bin_edges

    with open(os.path.join(input_savepath,'inputdata.pkl'),'wb') as file:
        pickle.dump(save_dict,file)

    print(f'Input data processed and saved to {input_savepath}')

if __name__ == "__main__":
    main()