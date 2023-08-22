import torch
import data_plotter
import utils.input_provider
import utils.prediction
import utils.dict_worker


def optimize(model, optimizer, data, batch_size, data_dim, out_dim, criterion, pred_len, seq_len):
    # x, y = utils_ba.get_single_pred(data, pred_len, seq_len, data_dim, out_dim)
    x, y = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim)
    zeros = torch.zeros_like(x[0:batch_size, :, :])
    NONE = None  # torch.zeros((1, 1, 1))

    def split_prediction_single():
        assert x.size(0) % batch_size == 0, "please choose a batch size that is a divisor of the total batch number"
        for i in range((x.size(0) // batch_size)):
            j = i

            def closure():

                optimizer.zero_grad()
                out = model.forward(x[batch_size * j: batch_size * (j + 1), :, :], NONE, zeros, NONE)
                out = out[:, :, data_dim - out_dim:]
                y_cut = y[batch_size * j: batch_size * (j + 1), :, :]
                loss = criterion(out, y_cut)
                loss.backward()
                return loss

            optimizer.step(closure)

    split_prediction_single()


def predict(model, data, train_data, batch_size, data_dim, out_dim, criterion, mae_crit, pred_len, seq_len, final=False):

    x, y = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim)
    out = utils.prediction.split_prediction(model, x, batch_size, dec_zero=True)
    out = out[:, :, data_dim - out_dim:]
    ret_dict = utils.dict_worker.pred_dict(out, y, criterion, mae_crit)

    x, y = utils.input_provider.get_input(train_data, pred_len, seq_len, data_dim, out_dim)
    out = utils.prediction.split_prediction(model, x, batch_size, dec_zero=True)
    out = out[:, :, data_dim - out_dim:]
    ret_dict.update(utils.dict_worker.pred_dict(out, y, criterion, mae_crit, train=True))

    if final:

        truth = torch.squeeze(data[0, :, :])
        x, _ = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, prediction=True)
        prediction = utils.prediction.split_prediction(model, x, batch_size, dec_zero=True)

        zeros = torch.zeros_like(data[0, :, :])
        prediction = prediction[0, :, :]
        zeros[seq_len: pred_len + seq_len, :] = prediction
        prediction = torch.squeeze(zeros)

        update_dict = data_plotter.log_pred_plots(prediction.cpu(), truth.cpu())

        ret_dict.update(update_dict)

        update_dict = data_plotter.get_error_over_time(out, y, out_dim)

        ret_dict.update(update_dict)

    return ret_dict
