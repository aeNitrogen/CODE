import torch
import data_plotter
import utils.input_provider
import utils.prediction
import utils.dict_worker
import utils.logging

alt = True


def optimize(model, optimizer, optimizer2, data, batch_size, data_dim, out_dim, criterion, pred_len, seq_len):

    x, x_dec, y = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, two_inputs=True)
    model2 = model.get_sub()

    if alt:
        temp = torch.zeros((x.size(0), x.size(1) + pred_len, x.size(2)))
        temp[:, :x.size(1), :] = x
        temp[:, x.size(1):, :-out_dim] = x_dec
        x = temp

    def split_prediction_single2(batch_size):
        # assert x.size(0) % batch_size == 0, "please choose a batch size that is a divisor of the total batch number"
        batch_add = 0
        if batch_size > x.size(0):
            batch_size = x.size(0)
        if x.size(0) % batch_size > 0:
            batch_add = 1
        for i in range((x.size(0) // batch_size) + batch_add):
            j = i

            def closure():
                start = batch_size * j
                end = min(batch_size * (j + 1), x.size(0))
                optimizer2.zero_grad()
                enc_slice = x[start: end, :, :]
                out = model2.forward(enc_slice)
                out = out[:, :, data_dim - out_dim:]
                y_cut = y[start: end, :, :]
                loss = criterion(out, y_cut)
                loss.backward()

                return loss

            optimizer2.step(closure)

    split_prediction_single2(batch_size)

    x = model2.forward(x)
    x = x[:, :, data_dim - out_dim:]
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
                start = batch_size * j
                end = min(batch_size * (j + 1), x.size(0))
                optimizer.zero_grad()
                enc_slice = x[start: end, :, :]
                dec_slice = x_dec[start: end, :, :]
                out = model.forward_train(enc_slice, None, dec_slice, None)
                y_cut = y[start: end, :, :]
                loss = criterion(out, y_cut)
                loss.backward()

                return loss

            optimizer.step(closure)

    split_prediction_single(batch_size)


def predict(model, data, train_data, batch_size, data_dim, out_dim, criterion, mae_crit, pred_len, seq_len,
            final=False, name=None):
    x, x_dec, y = utils.input_provider.get_input(train_data, pred_len, seq_len, data_dim, out_dim, two_inputs=True)

    if alt:
        temp = torch.zeros((x.size(0), x.size(1) + pred_len, x.size(2)))
        temp[:, :x.size(1), :] = x
        temp[:, x.size(1):, :-out_dim] = x_dec
        x = temp

    out = utils.prediction.split_prediction(model, x, batch_size, x_dec=x_dec)

    ret_dict = utils.dict_worker.pred_dict(out, y, criterion, mae_crit, train=True)

    x, x_dec, y = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, two_inputs=True)

    if alt:
        temp = torch.zeros((x.size(0), x.size(1) + pred_len, x.size(2)))
        temp[:, :x.size(1), :] = x
        temp[:, x.size(1):, :-out_dim] = x_dec
        x = temp

    out = utils.prediction.split_prediction(model, x, batch_size, x_dec=x_dec)

    ret_dict.update(utils.dict_worker.pred_dict(out, y, criterion, mae_crit))

    if final:
        assert name is not None, "please update iterator"

        ret_dict = utils.logging.log(ret_dict, out, y, out_dim, pred_len, name, data, data_dim, criterion, mae_crit)

        truth = torch.squeeze(data[0, :, data_dim - y.size(1):])
        x, x_dec, _ = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, prediction=True,
                                                     two_inputs=True)

        if alt:
            temp = torch.zeros((x.size(0), x.size(1) + pred_len, x.size(2)))
            temp[:, :x.size(1), :] = x
            temp[:, x.size(1):, :-out_dim] = x_dec
            x = temp

        prediction = utils.prediction.split_prediction(model, x, batch_size, x_dec=x_dec)
        prediction = torch.squeeze(prediction[0, :, :])
        update_dict = data_plotter.log_pred_plots(prediction.cpu(), truth.cpu())
        ret_dict.update(update_dict)

    return ret_dict
