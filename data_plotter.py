import torch
import matplotlib.pyplot as plt
import wandb
from scipy.signal import savgol_filter

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
    for i in range(pred_num):
        plt.figure(dpi=1200)
        plt.plot(pred[:, i], label="prediction " + i.__str__())
        plt.plot(target[:, i], label="target " + i.__str__())
        plt.legend()
        dict.update({"prediction " + i.__str__(): wandb.Image(plt)})
        plt.close()
        plt.clf()

    return dict


def loss_over_time(loss, min, max):

    plt.figure(dpi=1200)
    plt.plot(tocpu(loss), label="average error")
    plt.plot(tocpu(min), label="minimum error")
    plt.plot(tocpu(max), label="maximum error")
    plt.legend()
    dict = {
        "error_over_time": wandb.Image(plt)
    }
    plt.close()
    plt.clf()
    return dict


def smoof(target):
    print(type(target))
    smoofin = 5
    y = savgol_filter(tocpu(target), target.size(0), smoofin)
    return y


def tocpu(target):
    if type(target) == torch.Tensor:
        print("cast")
        return target.cpu()
    else:
        print("shpo")
        print(type(target))
        return target


def smoothing_old(target):
    smoo = 11
    assert smoo % 2 == 1, "please choose an odd smoothing factor"
    half = smoo // 2
    ret = torch.zeros_like(target[half: -half])
    for i in range(ret.size(0)):
        ret[i] = torch.mean(target[i:i + smoo])
    ret = torch.zeros_like(target)
    for i in range(target.size(0)):
        ret[i] = torch.mean(target[max(0, i - half): min(target.size(0), i + half + 1)])
    return ret


def cutup(target):
    percent = round(0.1 * target.size(0))
    ordered = torch.sort(target, dim=0)[0]
    ret = ordered[percent:-percent, :]
    # print(ret)
    print(type(ret))
    return ret[0, :], ret[ret.size(0) - 1, :]


def get_error_over_time(out, y, out_dim):
    e_o_t = out - y
    diff = torch.abs(e_o_t)
    e_o_t = torch.sum(diff, dim=2) / out_dim
    e_o_t_max, e_o_t_min = cutup(e_o_t)
    e_o_t = torch.sum(e_o_t, dim=0) / e_o_t.size(0)

    update_dict = loss_over_time(smoof(e_o_t), smoof(e_o_t_min), smoof(e_o_t_max))

    return update_dict




