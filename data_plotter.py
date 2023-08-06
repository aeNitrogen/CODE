import torch
import matplotlib.pyplot as plt
import wandb

plt.style.use('dark_background')


def plot_tensor(tensor, batch_number):
    plt.figure(dpi=1200)
    plotted = torch.transpose(tensor[batch_number], 0, 1)
    for i in range(plotted.size()[0]):

        plt.plot(plotted[i], label=i, linewidth=0.1)
    plt.legend()
    plt.show()


def plot_loss(loss_array, cut=True):
    if cut:
        loss_array = [1 if l_ > 1 else l_ for l_ in loss_array]
    plt.figure(dpi=1200)
    plt.plot(loss_array, label="loss", linewidth=0.1)
    plt.legend()
    plt.show()


def plot_diff(pred, target):
    plt.figure(dpi=1200)
    plt.plot(pred, label="prediction", linewidth=0.1)
    plt.plot(target, label="target", linewidth=0.1)
    plt.legend()
    plt.show()


def plot_diff_all(pred, target):
    for i in range(pred.size()[2]):
        plot_diff(pred[0, :, i], target[0, :, i])

def log_pred_plots(pred, target):
    pred_num = pred.size(1)
    dict = {}
    cols = []
    for i in range(pred_num):
        plt.figure(dpi=1200)
        plt.plot(pred[:, i], label="prediction " + i.__str__())
        plt.plot(target[:, i], label="target " + i.__str__())
        plt.legend()
        dict.update({"prediction " + i.__str__(): wandb.Image(plt)})
        plt.clf()




    return dict

