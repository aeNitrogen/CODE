import torch
import matplotlib.pyplot as plt
import wandb
from scipy.signal import savgol_filter
# plt.style.use('dark_background')

dpi=300

def plot_tensor(tensor, batch_number):
    plt.figure(dpi=dpi)
    plotted = torch.transpose(tensor[batch_number], 0, 1)
    for i in range(plotted.size()[0]):

        plt.plot(plotted[i], label=i, linewidth=0.1)
    plt.legend()
    plt.show()


def plot_loss(loss_array, cut=True):
    if cut:
        loss_array = [1 if l_ > 1 else l_ for l_ in loss_array]
    plt.figure(dpi=dpi)
    plt.plot(loss_array, label="loss", linewidth=0.1)
    plt.legend()
    plt.show()


def plot_diff(pred, target):
    plt.figure(dpi=dpi)
    plt.plot(pred, label="prediction", linewidth=0.1)
    plt.plot(target, label="target", linewidth=0.1)
    plt.legend()
    plt.show()


def plot_diff_all(pred, target):
    for i in range(pred.size()[2]):
        plot_diff(pred[0, :, i], target[0, :, i])


def log_pred_plots(pred, target, alt=False, start=150, pred_len=750):
    # print(pred_len)
    if pred_len < 750:  # cheetah or maze dataset
        start = 60
        # print("shortened")
    pred_num = pred.size(1)
    dict = {}
    x_t = range(target.size(0))
    x_p = x_t[start:start + pred_len]
    # print(len(x_t))
    # print(len(x_p))
    for i in range(pred_num):
        plt.figure(dpi=dpi)
        plt.plot(x_p, pred[:, i], label="prediction " + i.__str__())
        plt.plot(x_t, target[:, i], label="target " + i.__str__())
        plt.legend()
        if alt:
            dict.update({"alt prediction " + i.__str__(): wandb.Image(plt)})
        else:
            dict.update({"prediction " + i.__str__(): wandb.Image(plt)})
        plt.close()
        plt.clf()

    return dict


def loss_over_time(loss, min, max):

    plt.figure(dpi=dpi)
    plt.plot(tocpu(loss), label="average error")
    plt.plot(tocpu(min), label="minimum error")
    plt.plot(tocpu(max), label="maximum error")
    plt.legend()
    dict = {
        "error_over_time_old": wandb.Image(plt)
    }
    plt.close()
    plt.clf()
    return dict


def loss_over_time_new(x, acc_loss, window_loss, alt=False):

    # print(acc_loss)
    # print(window_loss)
    plt.figure(dpi=dpi)
    if alt:
        add = "alt "
    else:
        add = ""
    plt.plot(x, acc_loss, label="average error accumulated")
    plt.plot(x, window_loss, label="average error windowed")
    plt.legend()
    dict = {
        add + "error_over_time": wandb.Image(plt)
    }
    plt.close()
    plt.clf()
    return dict


def smoof(target):
    # print(type(target))
    smoofin = 5
    y = savgol_filter(tocpu(target), target.size(0), smoofin)
    return y


def tocpu(target):
    if type(target) == torch.Tensor:
        return target.cpu()
    else:
        # print("shpo")
        # print(type(target))
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
    # print(type(ret))
    return ret[0, :], ret[ret.size(0) - 1, :]


def RMSE(x, y):
    crit = torch.nn.MSELoss()
    z = torch.sqrt(crit(x, y))
    return z


def get_error_over_time_old(out, y, out_dim):
    e_o_t = out - y
    diff = torch.abs(e_o_t)
    e_o_t = torch.sum(diff, dim=2) / out_dim
    e_o_t_min, e_o_t_max = cutup(e_o_t)
    e_o_t = torch.sum(e_o_t, dim=0) / e_o_t.size(0)

    update_dict = loss_over_time(smoof(e_o_t), smoof(e_o_t_min), smoof(e_o_t_max))

    return update_dict


def wandbTable(x, name):
    cols = [name]
    if type(x) == torch.Tensor:
        # data = x.tolist()
        data = [[x[i].item()] for i in range(len(x))]
    else:
        data = [[x[i]] for i in range(len(x))]
    table = wandb.Table(columns=cols, data=data)
    return table


def create_tab(x, acc, window, alt=False):
    steps = wandbTable(x, "steps")
    rmse_acc = wandbTable(acc, "RMSE_over_time")
    rmse_wind = wandbTable(window, "RMSE_ot_windowed")
    if alt:
        add = "alt "
    else:
        add = ""
    dict = {
        add + "steps" : steps,
        add + "RMSE_over_time": rmse_acc,
        add + "RMSE_ot_windowed": rmse_wind
    }
    return dict


def get_error_over_time(out, y, out_dim, alt=False, normalized=False):
    window_size = 50
    # crit = torch.sqrt(torch.nn.MSELoss())
    crit = RMSE
    mse_window = []
    mse_acc = []
    x = []
    for i in range(out.size(1) // window_size):
        start = i * window_size
        end = (i + 1) * window_size
        mse_window.append(crit(out[:, start:end, :], y[:, start:end, :]).cpu())
        mse_acc.append(crit(out[:, :end, :], y[:, :end, :]).cpu())
        x.append((i + 1) * window_size)
    mse_acc = tocpu(mse_acc)
    mse_window = tocpu(mse_window)
    update_dict = loss_over_time_new(x, mse_acc, mse_window, alt=alt)
    update_dict.update(create_tab(x, mse_acc, mse_window, alt=alt))

    return update_dict
