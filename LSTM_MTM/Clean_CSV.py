import numpy as np
import warnings
import ast
warnings.filterwarnings('ignore')
import ast

import pandas as pd
import os

def clean_csv(df,columns):

    if len(columns) != 1:

        label_list = list(df.columns.values)[0:3]
        df.rename(columns={label_list[0]: 'Time'}, inplace=True)
        df.rename(columns={label_list[1]: 'Volume'}, inplace=True)
        df.rename(columns={label_list[2]: 'Gammatilde'}, inplace=True)
        columns = list(df.columns.values)[1:3]

        for column in columns:
            df[column] = df[column].apply(lambda x: ', '.join(x.split())) # standardiz the seperator
            df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df[column] = df[column].apply(lambda x: np.array([float(i) for i in x]))
                
    else:
        label_list = list(df.columns.values)[0:2]
        df.rename(columns={label_list[0]: 'Time'}, inplace=True)
        df.rename(columns={label_list[1]: 'Volume'}, inplace=True)
        columns = list(df.columns.values)[1:2]
        for column in columns:
            df[column] = df[column].apply(lambda x: ', '.join(x.split())) # standardiz the seperator
            df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df[column] = df[column].apply(lambda x: np.array([float(i) for i in x]))

        # for column in columns:
        #     df[column] = df[column].str.replace('[','').str.replace(']','').str.split(' ')
        #     df[column] = df[column].apply(lambda x: [i for i in x if i != ''])
        #     try:
        #         df[column] = df[column].apply(lambda x: np.array([float(i) for i in x]))
        #     except:
        #         df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df