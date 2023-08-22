import torch


def pred_dict(out, y, criterion, mae_crit, train=False):
    loss = criterion(out, y)
    mae = mae_crit(out, y)

    rmse = torch.sqrt(loss)
    rmae = torch.sqrt(mae)

    if train:
        ret_dict = {
            'train_loss': loss,
            'train_rmse': rmse,
            'train_mae': mae,
            'train_rmae': rmae
        }
    else:
        ret_dict = {
            'validation_loss': loss,
            'rmse': rmse,
            'mae': mae,
            'rmae': rmae
        }

    return ret_dict
