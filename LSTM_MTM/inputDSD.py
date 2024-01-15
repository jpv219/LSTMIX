##############################################################################
### Prepare and save the input data for DSD training (Stirred mixer)
### Author: Fuyue Liang
### First commit: Oct 2023
##############################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import math
import ast
import os
import pickle

import input as ipt

input_savepath = '/home/fl18/Desktop/automatework/ML_casestudy/LSTM_SMX/LSTM_MTM/input_data/'

####################################################
# Function to drop out outliers
def filter_data(my_data):
    for i in range(len(my_data)):

        pre_fil_data = np.array(my_data[i])
        Q1, median, Q3 = np.percentile(pre_fil_data, [25, 50, 75])
        IQR = Q3 - Q1

        loval = Q1 - 1.5 * IQR
        hival = Q3 + 1.5 * IQR

        wiskhi = np.compress(pre_fil_data <= hival, pre_fil_data)
        wisklo = np.compress(pre_fil_data >= loval, pre_fil_data)
        fil_data = np.compress(loval <= pre_fil_data <= hival, pre_fil_data)
        actual_hival = np.max(wiskhi)
        actual_loval = np.min(wisklo)
    
    return fil_data

# Function to transform the data into required data type
def clean_csv(df,columns): # make the volume string can be callable
    for column in columns:
        df[column] = df[column].apply(lambda x: ', '.join(x.split())) # standardize the separator
        df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df[column] = df[column].apply(lambda x: np.array([float(i) for i in x]))
    return df

# Function to scale the Drop volume 
def scale_DropVolume(df):
    l_cap = (0.035/(9.80665*(998-824)))**0.5
    v_cap = (4/3) * math.pi * (l_cap/2)**3
    df['DropVolume'] = df['DropVolume'].apply(lambda x: np.log10(x / v_cap).astype('float32'))
    return df

# Function to fine the range of drop size among all the cases
def find_rng(data_dir, cases):
    
    size_max, size_min = [], []
    
    for case in cases:
        data = pd.read_csv(str(data_dir)+str(case)+'.csv')
        if not 'clean' in case:
            data = clean_csv(data,['DropVolume','Gammatilde'])
        else:
            data['DropVolume'] = data['DropVolume'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            data['DropVolume'] = data['DropVolume'].apply(lambda x: np.array([float(i) for i in x]))
        data = scale_DropVolume(data)
        # fil_data = filter_data(data['DropVolume'])

        max_list = [max(x) for x in data['DropVolume'] if len(x) > 0]
        min_list = [min(x) for x in data['DropVolume'] if len(x) > 0]

        size_max.append(max(max_list))
        size_min.append(min(min_list))
    
    rng_ceil = math.ceil(max(size_max))
    rng_floor = math.floor(min(size_min))
    print('The range of the bin should be: ('+ str(rng_floor)+','+ str(rng_ceil)+')')
    
    return rng_floor, rng_ceil

# Function to define the range of drop size and bin number
def create_mybins(start, end, num, closed:str):
    mybins = pd.interval_range(start=start, periods=num, end=end, closed=closed)
    bin_edges = [interval.left for interval in mybins]
    bin_edges.append(mybins[-1].right)
    return mybins, bin_edges

# Functions to find the bin index and count the number
def find_bin(value, bins):
    """ bins is a list of tuples, like [(0,20), (20, 40), (40, 60)],
        binning returns the smallest index i of bins so that
        bin[i][0] <= value < bin[i][1]
    """
    
    for i in range(0, len(bins)):
#         if bins[i].left <= value < bins[i].right:
#             return i
        if value in bins[i]:
            return i
    return -1

def count_drops(data_dir,cases,bin_num,mybins):
    drops_in_bins = []
    
    for case in cases:
        data = pd.read_csv(str(data_dir)+str(case)+'.csv')
        if not 'clean' in case:
            data = clean_csv(data,['DropVolume','Gammatilde'])
        else:
            data['DropVolume'] = data['DropVolume'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            data['DropVolume'] = data['DropVolume'].apply(lambda x: np.array([float(i) for i in x]))
        data = scale_DropVolume(data)
        
        one_case = []
        # add the counts for a drop size range
        for i in range(64,449):# time step index from 256,i.e.,(320,704)
            bin_counts = np.zeros(bin_num).astype('int')
            for value in data['DropVolume'][i]:
                bin_index = find_bin(value, mybins)
                bin_counts[bin_index] += 1
                
            one_case.append(bin_counts)
        one_case_arr = np.array(one_case)    
        drops_in_bins.append(one_case_arr)
    
    arr = np.array(drops_in_bins)
    
    return arr

# Function to transform binned data to PDF
def density_func_est(bin_edges, binned_data,leftmost, rightmost, cases):
    '''
    Transform the binned data into its density functions (and normalized)
    Input: in shape of (cases, times, bins)
    leftmost: smallest drop size to transformed
    rightmost: largest drop size to transformed
    '''
    # calculate density estimation
    bin_widths = np.diff(bin_edges[leftmost:rightmost+1])
    trun_binned_data = binned_data[:,:,leftmost:rightmost] 
    total_count = np.sum(trun_binned_data, axis=-1) # count the drop size in one time step

    normed_densts = []
    for i, _ in enumerate(cases):
        # density function estimation
        # est_denst = trun_binned_data[i,:,:] / (total_count[i,:, np.newaxis] * bin_widths)
        est_denst = np.where(total_count[i,:,np.newaxis] > 0, trun_binned_data[i,:,:] / (total_count[i,:, np.newaxis] * bin_widths),0)
        total_denst = np.sum(est_denst,axis=-1)
        # normalize density function
        normed_denst = []
        for j in range(len(est_denst)):
            arr = np.where(total_denst[j]>0, est_denst[j,:] / total_denst[j], 0)
            normed_denst.append(arr)
        normed_denst= np.array(normed_denst)             
        # normed_denst = np.array([est_denst[j,:] / total_denst[j] for j in range(len(est_denst))])# if not total_denst[j] == 0])
        normed_densts.append(normed_denst)
    
    # replace NaN with 0 due to all zeros bin's count
    normed_densts=np.array(normed_densts)
    normed_densts[np.isnan(normed_densts)] = 0
    
    return normed_densts


############################################# MAIN #######################################
def main():

    data_dir = '/home/fl18/Desktop/automatework/RNN_auto/APSdata/'
    svcases = ['Bi0001','Bi0004','Bi001','B05','B07','clean_5hz','clean_6hz','clean_7hz','clean_9hz','clean_10hz','B09','Bi1','Bi0002','clean_8hz']
    bin_num = 20

    rng_floor, rng_ceil = find_rng(data_dir, svcases)
    mybins, bin_edges = create_mybins(start=rng_floor, end=rng_ceil, num=bin_num, closed='left')
    drops_in_bins = count_drops(data_dir, svcases, bin_num, mybins)
    
    leftmost=5
    rightmost=15

    bin_edges_to_saved = bin_edges[leftmost:rightmost]
    with open(os.path.join(input_savepath,'svinputdataBinswC.pkl'),'wb') as file:
        pickle.dump(bin_edges_to_saved,file)

    transdata = density_func_est(bin_edges, drops_in_bins, leftmost, rightmost,svcases)

    DSD_data = np.transpose(transdata,(1,0,2)).astype('float32')

    # smoothing data
    smoothed_data = ipt.smoothing(DSD_data,'lowess',lowess_frac=0.06)

    # save the data
    with open(os.path.join(input_savepath,'svinputdataDSDwC.pkl'),'wb') as file:
        pickle.dump(smoothed_data,file)
    with open(os.path.join(input_savepath,'svinputdataDSDwC_0.pkl'),'wb') as file:
        pickle.dump(DSD_data,file)

    # reading the saved Nd and IA
    with open(os.path.join(input_savepath,'svinputdatawC.pkl'), 'rb') as file:
        Nd_IA_data = pickle.load(file)
    
    alldata_0 = np.concatenate((Nd_IA_data, DSD_data), axis=-1)
    alldata = np.concatenate((Nd_IA_data, smoothed_data), axis=-1)

    # save the all Nd IA and DSD
    with open(os.path.join(input_savepath,'svinputdataALLwC_0.pkl'),'wb') as file:
        pickle.dump(alldata_0,file)
    with open(os.path.join(input_savepath,'svinputdataALLwC.pkl'),'wb') as file:
        pickle.dump(alldata,file)


if __name__ == "__main__":
    main()