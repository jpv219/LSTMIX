import torch
import modeltrain_LSTM as trn
from modeltrain_LSTM import Window_data


def main():

    ####### WINDOW DATA ########

    ## Windowing hyperparameters
    steps_in, steps_out = 50, 50
    stride = 1

    windowed_data = trn.windowing(steps_in,steps_out,stride)

    ## Extracting from named tuple
    X_train = windowed_data.X_train.to(torch.float32)
    y_train = windowed_data.y_train.to(torch.float32)
    X_val = windowed_data.X_val.to(torch.float32)
    y_val = windowed_data.y_val.to(torch.float32)

######## SAVING ALL RELEVANT DATA ########

    trn.saving_data(windowed_data, hp=None, model_choice='DMS')

if __name__ == "__main__":
    main()