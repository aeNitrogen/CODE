import torch
import data_plotter
import utils.input_provider
import utils.prediction
import utils.dict_worker
import data_loader
import utils.logging


def optimize(model, optimizer, data, batch_size, data_dim, out_dim, criterion, pred_len, seq_len, attn=False):

    x, x_dec, y = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, two_inputs=True)

    def split_prediction_single(batch_size):
        # assert x.size(0) % batch_size == 0, "please choose a batch size that is a divisor of the total batch number"
        batch_add = 0
        if batch_size > x.size(0):
            batch_size = x.size(0)
        if x.size(0) % batch_size > 0:
            batch_add = 1
        for i in range((x.size(0) // batch_size) + batch_add):
            j = i

            def closure():

                optimizer.zero_grad()
                start = batch_size * j
                end = min(batch_size * (j + 1), x.size(0))
                enc_slice = x[start: end, :, :]
                dec_slice = x_dec[start: end, :, :]
                if attn:
                    out, a = model.forward(enc_slice, None, dec_slice, None)
                else:
                    out = model.forward(enc_slice, None, dec_slice, None)
                y_cut = y[start: end, :, :]
                loss = criterion(out, y_cut)
                loss.backward()

                return loss

            optimizer.step(closure)

    split_prediction_single(batch_size)


def predict(model, data, train_data, batch_size, data_dim, out_dim, criterion, mae_crit, pred_len, seq_len,
            final=False, attn=False, name=None):
    x, x_dec, y = utils.input_provider.get_input(train_data, pred_len, seq_len, data_dim, out_dim, two_inputs=True)
    out = utils.prediction.split_prediction(model, x, batch_size, x_dec=x_dec, attn=attn)
    if attn:
        out = out[0]
    ret_dict = utils.dict_worker.pred_dict(out, y, criterion, mae_crit, train=True)

    x, x_dec, y = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, two_inputs=True)
    out = utils.prediction.split_prediction(model, x, batch_size, x_dec=x_dec, attn=attn)
    if attn:
        out = out[0]
    ret_dict.update(utils.dict_worker.pred_dict(out, y, criterion, mae_crit))

    if final:
        assert name is not None, "please update iterator"

        ret_dict = utils.logging.log(ret_dict, out, y, out_dim, pred_len, name, data, data_dim, criterion, mae_crit)

        truth = torch.squeeze(data[0, :, data_dim - y.size(1):])

        x, x_dec, _ = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, prediction=True,
                                                     two_inputs=True)
        if attn:
            prediction, attentions = utils.prediction.split_prediction(model, x, batch_size, x_dec=x_dec, attn=attn)
        else:
            prediction = utils.prediction.split_prediction(model, x, batch_size, x_dec=x_dec)

        prediction = torch.squeeze(prediction[0, :, :])
        update_dict = data_plotter.log_pred_plots(prediction.cpu(), truth.cpu(), start=150, pred_len=pred_len)
        ret_dict.update(update_dict)

    return ret_dict
