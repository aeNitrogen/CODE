import torch


def predictor(model, x_enc, x_dec, start, end, lstm=False):
    if x_dec is None:
        return model.forward(x_enc[start:end, :, :])
    else:
        if lstm:
            return model.forward(x_enc[start:end, :, :], x_dec[start:end, :, :])
        else:
            return model.forward(x_enc[start:end, :, :], None, x_dec[start:end, :, :], None)


def split_prediction(model, x, batch_size, x_dec=None, dec_zero=False, attn=False, lstm=False):
    """
    :param model:       model to apply forward operation to
    :param x:           encoder data
    :param batch_size:  batch size
    :param x_dec:       decoder data
    :param dec_zero:    option to generate decoder zeros in the shape of x
    :return:            prediction of model
    """
    # assert x.size(0) % batch_size == 0, "please choose a batch size that is a divisor of the total batch number"
    if batch_size > x.size(0):
        batch_size = x.size(0)
    batch_add = 0
    if x.size(0) % batch_size > 0:
        batch_add = 1

    if dec_zero:
        x_dec = torch.zeros_like(x)

    if attn:
        y, a = predictor(model, x, x_dec, 0, batch_size)
    else:
        y = predictor(model, x, x_dec, 0, batch_size, lstm=lstm)
    for i in range((x.size(0) // batch_size) - 1 + batch_add):
        j = i + 1
        start = batch_size * j
        end = min(batch_size * (j + 1), x.size(0))
        if attn:
            y_t, a_t = predictor(model, x, x_dec, start, end)
            y = torch.cat((y, y_t), dim=0)
            a = torch.cat((a, a_t), dim=0)
        else:
            y = torch.cat((y, predictor(model, x, x_dec, start, end, lstm=lstm)), dim=0)

    if attn:
        return y, a
    return y
