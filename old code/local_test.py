import torch
import pickle
from os import path

basepath = path.dirname(__file__)

def load_pickled(name):
    # load_path = path + "\\" + name
    load_path = "DATA/" + name
    filepath = path.abspath(path.join(basepath,"..", load_path))
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        return data


def print_data(data_dict):
    print("Train Obs Shape", data_dict['train_obs'].shape)
    print("Train Act Shape", data_dict['train_act'].shape)
    print("Train Targets Shape", data_dict['train_targets'].shape)
    print("Test Obs Shape", data_dict['test_obs'].shape)
    print("Test Act Shape", data_dict['test_act'].shape)
    print("Test Targets Shape", data_dict['test_targets'].shape)
    # print("Normalizer", data_dict['normalizer'])


# print_data(load_pickled("CheetahNeuripsData.pkl"))
# print_data(load_pickled("mediumNeuripsData.pkl"))
print_data(load_pickled("SinMixData.pkl"))
exit()
