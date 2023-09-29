import torch
import data_plotter
import utils.input_provider
import utils.prediction
import utils.dict_worker
import data_loader


def log(ret_dict, out, y, out_dim, pred_len, name, data, data_dim, criterion, mae_crit):

    update_dict = data_plotter.get_error_over_time(out, y, out_dim)
    ret_dict.update(update_dict)

    out = data_loader.denormalizer(out, name)
    y = data_loader.denormalizer(y, name)
    den_data = data_loader.denormalizer(data[:, :, data_dim - y.size(2):], name)
    ret_dict.update(utils.dict_worker.pred_dict(out, y, criterion, mae_crit, denorm=True))

    update_dict = data_plotter.get_error_over_time(out, y, out_dim)
    ret_dict.update(update_dict)

    den_data = torch.squeeze(den_data[0, :, :])
    y = torch.squeeze(y[0, :, :])
    out = out[0, :, :]
    out = torch.squeeze(out)

    update_dict = data_plotter.log_pred_plots(out.cpu(), den_data.cpu(), alt=True, start=150, pred_len=pred_len)
    ret_dict.update(update_dict)
    return ret_dict