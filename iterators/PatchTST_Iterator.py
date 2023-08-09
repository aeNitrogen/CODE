import utils_ba


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
