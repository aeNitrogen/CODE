import pickle
import torch


def normalize(tensor, normalizer_):
    tensor = torch.tensor(tensor)
    normalizer_ = torch.tensor(normalizer_)
    batch_number = tensor.size()[0]
    norm_s = torch.repeat_interleave(torch.tensor(normalizer_[0])[None, :], torch.tensor([900]), dim=0)
    # values to be subtracted, extend to time
    norm_s = torch.repeat_interleave(norm_s[None, :], torch.tensor([batch_number]), dim=0)  # extendBatch
    norm_div = normalizer_[1][None, :][None, :]
    norm_div = torch.tensor(norm_div)

    result = (tensor - norm_s) / norm_div

    return result


def assign(data):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("DEBUG: Device: " + device)

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

    print("DEBUG: data loaded")

    train_obs_norm = normalize(train_obs, norm_observations)
    train_act_norm = normalize(train_act, norm_actions)
    train_input = torch.cat((train_act_norm, train_obs_norm), 2)

    test_obs_norm = normalize(test_obs, norm_observations)
    test_act_norm = normalize(test_act, norm_actions)
    test_input = torch.cat((test_act_norm, test_obs_norm), 2)

    optimize_train = train_input[:100, :, :].float().to(device="cuda:0", copy=True)
    optimize_test = test_input[:100, :, :].float().to(device=device, copy=True)
    final_train = train_input[100:, :, :].float().to(device=device, copy=True)
    final_test = test_input[100:, :, :].float().to(device=device, copy=True)

    print("DEBUG: data processed")

    return optimize_train, optimize_test, final_train, final_test


def load_pickled(name):
    # load_path = path + "\\" + name
    load_path = "DATA/" + name
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
        return data


def load(name):
    return assign(load_pickled(name))
