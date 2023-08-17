import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def clean_csv(df,columns):

    if len(columns) != 1:

        if columns[1] != 'Volume':
            label_list = list(df.columns.values)[0:3]
            df.rename(columns={label_list[0]: 'Time'}, inplace=True)
            df.rename(columns={label_list[1]: 'Volume'}, inplace=True)
            df.rename(columns={label_list[2]: 'Gammatilde'}, inplace=True)
            columns = list(df.columns.values)[1:3]

        for column in columns:
            df[column] = df[column].str.replace('[','').str.replace(']','').str.split(' ')
            df[column] = df[column].apply(lambda x: [i for i in x if i != ''])
            df[column] = df[column].apply(lambda x: np.array([float(i) for i in x]))
    else:
        for column in columns:
            df[column] = df[column].str.replace('[','').str.replace(']','').str.split(' ')
            df[column] = df[column].apply(lambda x: [i for i in x if i != ''])
            df[column] = df[column].apply(lambda x: np.array([float(i) for i in x]))
    return df