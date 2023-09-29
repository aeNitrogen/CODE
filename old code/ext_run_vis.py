import math
from os.path import isfile, join
from os import listdir
import torch
import pickle
from os import path
from numpy import load
from statistics import mean
print("start")

basepath = path.dirname(__file__)

criterion = torch.nn.MSELoss()
mae_crit = torch.nn.L1Loss()

crit = torch.nn.MSELoss()
zeros = torch.zeros((3,3))
ones = torch.ones((3,3))
ones[:,1 ] = 2
ones[:, 2] = 3

print(crit(zeros, ones))


def load_pickled(name):
    # load_path = path + "\\" + name
    load_path = "DATA/" + name
    filepath = path.abspath(path.join(basepath,"..", load_path))
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        return data

def load_run(name):
    load_path = "DATA/foreign_runs/CheetahTrue/AR-Transformer/" + name
    filepath = path.abspath(path.join(basepath, "..", load_path))
    data = load(filepath)
    lst = data.files
    t = None
    for item in lst:
        t = torch.tensor(data[item])

def calc_metrics(gt, tensors):
    mse = mean([criterion(gt, t).item() for t in tensors])
    mae = mean([mae_crit(gt, t).item() for t in tensors])
    print("RMSE " + math.sqrt(mse).__str__())
    print("MSE " + mse.__str__())
    print("MAE " + mae.__str__())

def calc_model(name):
    load_path = "DATA/foreign_runs/Maze/" + name
    filepath = path.abspath(path.join(basepath, "..", load_path))
    onlyfiles = [f for f in listdir(filepath) if (isfile(join(filepath, f)) and ("pred_mean" in f))]
    data = load(filepath + "/gt.npz")
    for item in data.files:
        gt = torch.tensor(data[item])
    tensors = [torch.tensor(load(filepath + "/" + run)[load(filepath + "/" + run).files[0]]) for run in onlyfiles]
    calc_metrics(gt, tensors)


#load_run("gt.npz")
#calc_model("AR-Transformer")
#calc_model("MTS3")

def calc_all():
    load_path = "DATA/foreign_runs/Maze/"
    filepath = path.abspath(path.join(basepath, "..", load_path))
    dirs = [f for f in listdir(filepath)]
    for f in dirs:
        print(f)
        calc_model(f)

# calc_all()
