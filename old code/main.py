# This is a sample Python script.
import data_loader
import data_plotter
import lstm
import transformer
import torch
import Nlinear
import wandb_interface
import numpy as np
import Dlinear

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
train_obs = None
train_act = None
train_targets = None
test_obs = None
test_act = None
test_targets = None
normalizer = None

norm_observations = None
norm_actions = None
norm_diff = None
norm_targets = None


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def assign(data):
    global train_obs
    global train_act
    global train_targets
    global test_obs
    global test_act
    global test_targets
    global normalizer

    global norm_observations
    global norm_actions
    global norm_diff
    global norm_targets

    train_obs = data.get('train_obs')
    train_act = data.get('train_act')
    train_targets = data.get('train_targets')
    test_obs = data.get('test_obs')
    test_act = data.get('test_act')
    test_targets = data.get('test_targets')
    normalizer = data.get('normalizer')

    norm_observations = normalizer.get('observations')
    norm_actions = normalizer.get('actions')
    norm_diff = normalizer.get('diff')
    norm_targets = normalizer.get('targets')

def normalize(tensor, normalizer_, graph):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("DEBUG: Device: " + device)
    # assert device == "cuda:0"
    device = torch.device(device)
    torch.set_default_device(device)
    batch_number = tensor.size()[0]
    # norm_sub = normalizer_[0][None, :][None, :]  # values to be subtracted
    norm_s = torch.repeat_interleave(torch.tensor(normalizer_[0], device=device)[None, :], torch.tensor([900],
                                                                                                        device=device),
                                     dim=0)  # values to be subtracted
    norm_s = torch.repeat_interleave(norm_s[None, :], torch.tensor([batch_number], device=device), dim=0)
    norm_div = normalizer_[1][None, :][None, :]
    norm_s = norm_s.to(device=device, copy=True)
    assert norm_s.device == device
    tensor = tensor.to(device=device, copy=True)
    assert tensor.device == device
    result = (tensor - norm_s) / norm_div
    if graph:
        data_plotter.plot_tensor(tensor, 100)
        data_plotter.plot_tensor(result, 100)
    return result
# makes no sense as of now
def normalize2(tensor, normalizer_, graph):
    batch_number = tensor.size()[0]
    norm_sub = normalizer_[0][None, :][None, :]
    norm_s = torch.repeat_interleave(torch.tensor(normalizer_[0])[None, :], torch.tensor([900]), dim=0)
    norm_s = torch.repeat_interleave(norm_s[None, :], torch.tensor([batch_number]), dim=0)
    norm_div = normalizer_[1][None, :][None, :]
    result = (tensor - norm_s) / norm_div
    if graph:
        data_plotter.plot_tensor(tensor, 100)
        data_plotter.plot_tensor(result, 100)
    return result


show_graphs = False
if __name__ == '__main__':

    data = data_loader.load_pickled('SinData.pkl')
    assign(data)
    # data_plotter.plot_tensor(train_obs, 0)
    train_obs_norm = normalize(train_obs, norm_observations, show_graphs)
    train_act_norm = normalize(train_act, norm_actions, show_graphs)
    train_input = torch.cat((train_act_norm, train_obs_norm), 2)
    test_obs_norm = normalize(test_obs, norm_observations, show_graphs)
    test_act_norm = normalize(test_act, norm_actions, show_graphs)
    test_input = torch.cat((test_act_norm, test_obs_norm), 2)
    test_targets_norm = normalize(test_targets, norm_targets, show_graphs)
    train_targets_norm = normalize(train_targets, norm_targets, show_graphs)

    # train_input = posEnc.fwd(train_input)
    # data_plotter.plot_tensor(pos - train_input, 0)
    # test_input = posEnc.fwd(test_input)
    # test_targets_norm = posEnc.fwd(torch.cat((test_act_norm, test_targets_norm), 2))[:, :, -9:]
    # train_targets_norm = posEnc.fwd(torch.cat((train_act_norm, train_targets_norm), 2))[:, :, -9:]
    # data_plotter.plot_tensor(test_targets_norm, 0)

    # print(norm_observations)
    # print(test_targets.size())
    # print(test_input.size())
    # print(train_input.dtype)

    # transformer.get_blocks(train_input, train_targets)

    # lstm2.calc(train_input.float(), train_targets_norm.float(), test_input.float(), test_targets_norm.float())
    optimize_train = train_input[:100, :, :].float()
    optimize_test = test_input[:100, :, :].float()
    final_train = train_input[100:, :, :].float()
    final_test = test_input[100:, :, :].float()

    architecture = "lstm"

    n_hidden_enc = 128 * 8
    n_hidden_dec = 128 * 8
    d_model = 32
    dropout = 0.1
    n_heads = 4
    overlap = 5

    epochs = 400
    opt = "adam"
    seq_len = 150
    pred_len = 500
    lr_start = 0.003
    lr_stop = 0.01
    diff_lr = lr_stop - lr_start
    lr = 0.2
    pred_res = None
    target_res = None
    x = np.arange(0.1, 1.01, 0.1)
    # x = np.append(np.arange(0.01, 0.1, 0.01), x)
    # x = np.append(np.arange(0, 0.01, 0.001), x)
    for i in x:
        lr = i
        epochs = 400
        # result = Nlinear.calc(optimize_train, optimize_test, seq_len=seq_len, pred_len=pred_len,
        #                       epochs=epochs, lr=lr, opt=opt, individual=True)
        if architecture == "lstm":

            result, pred_res, target_res = lstm.calc(optimize_train, optimize_test, lr=lr, d_model=d_model,
                                                     epochs=epochs, pred_len=pred_len, seq_len=seq_len)
        elif architecture == "nlinear-individiual":
            result, pred_res, target_res = Nlinear.calc(optimize_train, optimize_test, seq_len=seq_len,
                                                        pred_len=pred_len, epochs=epochs, lr=lr, opt=opt,
                                                        individual=True)
        elif architecture == "nlinear-non-individiual":
            result, pred_res, target_res = Nlinear.calc(optimize_train, optimize_test, seq_len=seq_len,
                                                        pred_len=pred_len, epochs=epochs, lr=lr, opt=opt,
                                                        individual=False)
        elif architecture == 'dlinear-individual':
            result, pred_res, target_res = Dlinear.calc(optimize_train, optimize_test, seq_len=seq_len,
                                                        pred_len=pred_len, epochs=epochs, lr=lr, opt=opt,
                                                        individual=True)

        elif architecture == 'dlinear-non-individual':
            result, pred_res, target_res = Dlinear.calc(optimize_train, optimize_test, seq_len=seq_len,
                                                        pred_len=pred_len, epochs=epochs, lr=lr, opt=opt,
                                                        individual=False)

        elif "trans" in architecture:

            result = transformer.calc(optimize_train, optimize_test, seq_len=seq_len, pred_len=pred_len,
                                  epochs=epochs, lr=lr, opt=opt, n_hidden_dec=n_hidden_dec, n_hidden_enc=n_hidden_enc,
                                  d_model=d_model, dropout=dropout, dec_overlap=overlap, n_heads=n_heads)

        wandb_interface.log(result, architecture=architecture, seq_len=seq_len, pred_len=pred_len, epochs=epochs,
                            info="hyperparametertuning-slirm", opt=opt, lr=lr, hidden_dec=n_hidden_dec,
                            hidden_enc=n_hidden_enc, d_model=d_model, dropout=dropout, overlap=overlap, n_heads=n_heads,
                            pred=pred_res, target=target_res)
    # transformer.calc(train_input.float(), test_input.float())
    # transformer.calc(train_input.float(), train_targets_norm.float(), test_input.float(), test_targets_norm.float())
    # --lstm.calc(train_input, train_targets, test_input, test_targets)

    # [batch, time , feature] [750, 900, 9/4/13]
    # data_plotter.plot_loss(result)
