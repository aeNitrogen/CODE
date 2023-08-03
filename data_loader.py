import pickle
import torch


# from unipath import Path


# path = Path('E:\\uni kram\\BA_NEW\\DATA')

def normalize(tensor, normalizer_):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.set_default_device(device)
    batch_number = tensor.size()[0]
    norm_s = torch.repeat_interleave(torch.tensor(normalizer_[0], device=device)[None, :], torch.tensor([900],
                                                                                                        device=device),
                                     dim=0)  # values to be subtracted, extend to time
    norm_s = torch.repeat_interleave(norm_s[None, :], torch.tensor([batch_number], device=device), dim=0)  # extendBatch
    norm_div = normalizer_[1][None, :][None, :]
    norm_div = torch.tensor(norm_div)

    norm_div = norm_div.to(device=device, copy=True)
    norm_s = norm_s.to(device=device, copy=True)
    tensor = tensor.to(device=device, copy=True)
    result = (tensor - norm_s) / norm_div

    return result


def assign(data):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("DEBUG: Device: " + device)
    torch.set_default_device(device)

    train_obs = data.get('train_obs')
    train_act = data.get('train_act')
    train_targets = data.get('train_targets')
    test_obs = data.get('test_obs')
    test_act = data.get('test_act')
    test_targets = data.get('test_targets')
    normalizer = data.get('normalizer')

    norm_observations = normalizer.get('observations')
    norm_actions = normalizer.get('actions')
    norm_diff = normalizer.get('diff')
    norm_targets = normalizer.get('targets')

    train_obs_norm = normalize(train_obs, norm_observations)
    train_act_norm = normalize(train_act, norm_actions)
    train_input = torch.cat((train_act_norm, train_obs_norm), 2)
    test_obs_norm = normalize(test_obs, norm_observations)
    test_act_norm = normalize(test_act, norm_actions)
    test_input = torch.cat((test_act_norm, test_obs_norm), 2)
    test_targets_norm = normalize(test_targets, norm_targets)
    train_targets_norm = normalize(train_targets, norm_targets)

    optimize_train = train_input[:100, :, :].float()
    optimize_test = test_input[:100, :, :].float()
    final_train = train_input[100:, :, :].float()
    final_test = test_input[100:, :, :].float()

    return optimize_train, optimize_test, final_train, final_test


def load_pickled(name):
    # load_path = path + "\\" + name
    load_path = "DATA/" + name
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
        return data


def load(name):
    return assign(load_pickled(name))
