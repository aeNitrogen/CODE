import torch
import data_plotter
import utils.input_provider
import utils.prediction
import utils.dict_worker

def optimize(model, optimizer, data, batch_size, data_dim, out_dim, criterion, pred_len, seq_len):

    x, x_dec, y = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, two_inputs=True)

    def split_prediction_single():
        assert x.size(0) % batch_size == 0, "please choose a batch size that is a divisor of the total batch number"
        for i in range((x.size(0) // batch_size)):
            j = i

            def closure():

                optimizer.zero_grad()
                enc_slice = x[batch_size * j: batch_size * (j + 1), :, :]
                dec_slice = x_dec[batch_size * j: batch_size * (j + 1), :, :]
                out = model.forward(enc_slice, None, dec_slice, None)
                y_cut = y[batch_size * j: batch_size * (j + 1), :, :]
                loss = criterion(out, y_cut)
                loss.backward()

                return loss

            optimizer.step(closure)

    split_prediction_single()


def predict(model, data, train_data, batch_size, data_dim, out_dim, criterion, mae_crit, pred_len, seq_len, final=False):

    x, x_dec, y = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, two_inputs=True)
    out = utils.prediction.split_prediction(model, x, batch_size, x_dec=x_dec)

    ret_dict = utils.dict_worker.pred_dict(out, y, criterion, mae_crit)

    x, x_dec, y = utils.input_provider.get_input(train_data, pred_len, seq_len, data_dim, out_dim, two_inputs=True)
    out = utils.prediction.split_prediction(model, x, batch_size, x_dec=x_dec)

    ret_dict.update(utils.dict_worker.pred_dict(out, y, criterion, mae_crit, train=True))

    if final:
        truth = torch.squeeze(data[0, :, :])
        x, x_dec, _ = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, prediction=True,
                                                     two_inputs=True)
        prediction = utils.prediction.split_prediction(model, x, batch_size, x_dec=x_dec)
        zeros = torch.zeros_like(data[0, :, :])
        prediction = prediction[0, :, :]
        zeros[seq_len: pred_len + seq_len, data_dim - out_dim:] = prediction
        prediction = torch.squeeze(zeros)
        update_dict = data_plotter.log_pred_plots(prediction.cpu(), truth.cpu())
        ret_dict.update(update_dict)
        update_dict = data_plotter.get_error_over_time(out, y, out_dim)
        ret_dict.update(update_dict)
    return ret_dict
