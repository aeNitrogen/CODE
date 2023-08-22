import torch


def predictor(model, x_enc, x_dec, start, end):
    if x_dec is None:
        return model.forward(x_enc[start:end, :, :])
    else:
        return model.forward(x_enc[start:end, :, :], None, x_dec[start:end, :, :], None)


def split_prediction(model, x, batch_size, x_dec=None, dec_zero=False):
    """
    :param model:       model to apply forward operation to
    :param x:           encoder data
    :param batch_size:  batch size
    :param x_dec:       decoder data
    :param dec_zero:    option to generate decoder zeros in the shape of x
    :return:            prediction of model
    """
    assert x.size(0) % batch_size == 0, "please choose a batch size that is a divisor of the total batch number"

    if dec_zero:
        x_dec = torch.zeros_like(x)

    y = predictor(model, x, x_dec, 0, batch_size)
    for i in range((x.size(0) // batch_size) - 1):
        j = i + 1
        start = batch_size * j
        end = batch_size * (j + 1)
        y = torch.cat(y, predictor(model, x, x_dec, start, end))

    return y
