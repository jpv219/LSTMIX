# Loading, Cleaning Volume-Gamma-Nd dataframes
import pandas as pd
import Clean_CSV
import os
import configparser

## Env. variables ##

## Setting up paths globally

config_paths = configparser.ConfigParser()
package_dir = os.path.dirname(os.path.abspath(__file__))
config_paths.read(os.path.join(package_dir, 'config/config_paths.ini'))

global_dir = config_paths['Path']['raw_data']

### Generate dataframe with Gamma and drop volume values
def extract_GVol(elem):
    csv_file = os.path.join(global_dir,f'{elem}_GVol.csv')
    df = pd.read_csv(csv_file)
    df = Clean_CSV.clean_csv(df,list(df.columns.values)[1:3])
    return df

### Generate dataframe with number of drops
def extract_Nd(elem):
    Nd_csv_file = os.path.join(global_dir,'Nd',f'{elem}_dnum_corr.csv')
    df = pd.read_csv(Nd_csv_file,header=None)
    label_list = list(df.columns.values)
    df.rename(columns={label_list[0]: 'Ndrops'}, inplace = True)
    df['Time'] = df.apply(lambda row: row.name*0.005,axis=1)
    df = df[['Time','Ndrops']]
    return df

### Generate dataframe with interfacial area
def extract_IA(elem):
    IA_csv_file = os.path.join(global_dir,'IA',f'{elem}_IA_corr.csv')
    df = pd.read_csv(IA_csv_file,header=None)
    label_list = list(df.columns.values)
    df.rename(columns={label_list[0]: 'IA'}, inplace = True)
    df['Time'] = df.apply(lambda row: row.name*0.005,axis=1)
    df = df[['Time','IA']]
    return df

### Generate dataframe with drop volumes
def extract_Vol(elem):
    csv_file = os.path.join(global_dir,f'{elem}_Vol.csv')
    df = pd.read_csv(csv_file)
    df = Clean_CSV.clean_csv(df,list(df.columns.values)[1:2])
    return df



