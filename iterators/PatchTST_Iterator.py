import torch
import data_plotter
import utils.input_provider
import utils.prediction
import utils.dict_worker
import utils.logging


def optimize(model, optimizer, data, batch_size, data_dim, out_dim, criterion, pred_len, seq_len):
    # x, y = utils_ba.get_single_pred(data, pred_len, seq_len, data_dim, out_dim)
    x, y = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim)

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
                out = model.forward(x[start: end, :, :])
                out = out[:, :, data_dim - out_dim:]
                y_cut = y[start: end, :, :]
                loss = criterion(out, y_cut)
                loss.backward()
                return loss

            optimizer.step(closure)

    split_prediction_single(batch_size)


def predict(model, data, train_data, batch_size, data_dim, out_dim, criterion, mae_crit, pred_len, seq_len,
            final=False, name=None):

    x, y = utils.input_provider.get_input(train_data, pred_len, seq_len, data_dim, out_dim)

    out = utils.prediction.split_prediction(model, x, batch_size)
    out = out[:, :, data_dim - out_dim:]
    ret_dict = utils.dict_worker.pred_dict(out, y, criterion, mae_crit, train=True)

    x, y = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim)
    out = utils.prediction.split_prediction(model, x, batch_size)
    out = out[:, :, data_dim - out_dim:]
    ret_dict.update(utils.dict_worker.pred_dict(out, y, criterion, mae_crit))

    if final:
        assert name is not None, "please update iterator"

        ret_dict = utils.logging.log(ret_dict, out, y, out_dim, pred_len, name, data, data_dim, criterion, mae_crit)

        truth = torch.squeeze(data[0, :, data_dim - y.size(1):])
        x, _ = utils.input_provider.get_input(data, pred_len, seq_len, data_dim, out_dim, prediction=True)
        prediction = utils.prediction.split_prediction(model, x, batch_size)
        # print("wat")
        prediction = prediction[:, :, data_dim - out_dim:]  # maybe
        prediction = torch.squeeze(prediction[0, :, :])
        # print(prediction.size())
        # print(truth.size())
        update_dict = data_plotter.log_pred_plots(prediction.cpu(), truth.cpu(), pred_len=pred_len)
        ret_dict.update(update_dict)
        # print("wat")
    return ret_dict
