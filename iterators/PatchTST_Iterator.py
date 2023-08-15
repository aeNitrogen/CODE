import utils_ba
import torch
import data_plotter

def optimize(model, optimizer, data, batch_size, data_dim, out_dim, criterion, pred_len, seq_len):

    x, y = utils_ba.get_single_pred(data, pred_len, seq_len, data_dim, out_dim)

    def split_prediction_single():

        assert x.size(0) % batch_size == 0, "please choose a batch size that is a divisor of the total batch number"
        for i in range((x.size(0) // batch_size)):
            j = i

            def closure():
                out = model.forward(x[batch_size * j: batch_size * (j + 1), :, :])
                out = out[:, :, data_dim - out_dim:]
                y_cut = y[batch_size * j: batch_size * (j + 1), :, :]
                loss = criterion(out, y_cut)
                loss.backward
                return loss
            optimizer.step(closure)

    split_prediction_single()


def predict(model, data, batch_size, data_dim, out_dim, criterion, mae_crit, pred_len, seq_len, final=False):
    x, y = utils_ba.get_single(data, pred_len, seq_len, data_dim, out_dim)

    # out = self.model.forward(x)
    out = utils_ba.split_prediction_single(model, x, batch_size)

    out = out[:, :, data_dim - out_dim:]
    loss = criterion(out, y)
    mae = mae_crit(out, y)

    rmse = torch.sqrt(loss)
    rmae = torch.sqrt(mae)

    ret_dict = {
        'validation_loss': loss,
        'rmse': rmse,
        'mae': mae,
        'rmae': rmae
    }

    if final:

        truth = torch.squeeze(data[0, :, :])
        x, _ = utils_ba.get_single_pred(data, pred_len, seq_len, data_dim, out_dim)
        prediction = utils_ba.split_prediction_single(model, x, batch_size)

        zeros = torch.zeros_like(data[0, :, :])
        prediction = prediction[0, :, :]
        zeros[seq_len: pred_len + seq_len, :] = prediction
        prediction = torch.squeeze(zeros)

        update_dict = data_plotter.log_pred_plots(prediction.cpu(), truth.cpu())

        ret_dict.update(update_dict)

        update_dict = data_plotter.get_error_over_time(out, y, out_dim)

        ret_dict.update(update_dict)

    return ret_dict
