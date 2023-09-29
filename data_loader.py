import pickle
import torch


def normalize(tensor, normalizer_):
    tensor = torch.tensor(tensor)
    normalizer_ = torch.tensor(normalizer_)
    batch_number = tensor.size(0)
    norm_s = torch.repeat_interleave(torch.tensor(normalizer_[0])[None, :], tensor.size(1), dim=0) # was torch tensor [900]
    # values to be subtracted, extend to time
    norm_s = torch.repeat_interleave(norm_s[None, :], torch.tensor([batch_number]), dim=0)  # extendBatch
    norm_div = normalizer_[1][None, :][None, :]
    norm_div = torch.tensor(norm_div)

    result = (tensor - norm_s) / norm_div

    return result


def denormalize_(tensor, normalizer_):
    normalizer_ = torch.tensor(normalizer_)
    batch_number = tensor.size(0)
    norm_s = torch.repeat_interleave(torch.tensor(normalizer_[0])[None, :], tensor.size(1), dim=0) # was torch tensor [900]
    # values to be subtracted, extend to time
    norm_s = torch.repeat_interleave(norm_s[None, :], torch.tensor([batch_number]), dim=0)  # extendBatch
    norm_div = normalizer_[1][None, :][None, :]
    result = (tensor * norm_div) + norm_s
    return result


def normalize_(tensor, normalizer_):
    normalizer_ = torch.tensor(normalizer_)
    batch_number = tensor.size(0)
    norm_s = torch.repeat_interleave(torch.tensor(normalizer_[0])[None, :], tensor.size(1), dim=0) # was torch tensor [900]
    # values to be subtracted, extend to time
    norm_s = torch.repeat_interleave(norm_s[None, :], torch.tensor([batch_number]), dim=0)  # extendBatch
    norm_div = normalizer_[1][None, :][None, :]
    result = (tensor - norm_s) / norm_div
    return result


def assign(data, name):

    if name in ["SinMixData.pkl", "CheetahNeuripsData.pkl", "mediumNeuripsData.pkl"]:
        print("DEBUG: no normalization applied")
        norm = False
    elif name in ["SinData.pkl"]:
        print("DEBUG: normalization applied")
        norm = True
    else:
        assert False, "unsupported dataset"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("DEBUG: Device: " + device)

    train_obs = data.get('train_obs')
    train_act = data.get('train_act')
    train_targets = data.get('train_targets')
    test_obs = data.get('test_obs')
    test_act = data.get('test_act')
    test_targets = data.get('test_targets')
    normalizer = data.get('normalizer')

    # normalized(train_obs)
    # print(train_obs[0:, :, :])
    # print(test_obs[0:, :, :])

    norm_observations = normalizer.get('observations')
    norm_actions = normalizer.get('actions')
    norm_diff = normalizer.get('diff')
    norm_targets = normalizer.get('targets')

    # print(norm_observations)
    # print(train_obs[0, :10, :])
    print("DEBUG: data loaded")

    if norm:
        train_obs_norm = normalize(train_obs, norm_observations)
        train_act_norm = normalize(train_act, norm_actions)
        train_input = torch.cat((train_act_norm, train_obs_norm), 2)
    else:
        train_input = torch.cat((train_act, train_obs), 2)

    if norm:
        test_obs_norm = normalize(test_obs, norm_observations)
        test_act_norm = normalize(test_act, norm_actions)
        test_input = torch.cat((test_act_norm, test_obs_norm), 2)
    else:
        test_input = torch.cat((test_act, test_obs), 2)

    # normalized(test_obs_norm)

    opt_train_end = 100
    opt_test_end = 100

    if name == "CheetahNeuripsData.pkl":
        opt_train_end = 100
        opt_test_end = 100
    elif name == "mediumNeuripsData.pkl":
        opt_train_end = 500
        opt_test_end = 100

    optimize_train = train_input[:opt_train_end, :, :].float().to(device="cuda:0", copy=True)
    optimize_test = test_input[:opt_test_end, :, :].float().to(device=device, copy=True)
    final_train = train_input[:, :, :].float().to(device=device, copy=True)
    final_test = test_input[:, :, :].float().to(device=device, copy=True)

    print("DEBUG: data processed")
    print(optimize_train.size())
    print(optimize_test.size())
    # print(optimize_train[0:, :, :])
    # print(optimize_test[0:, :, :])
    return optimize_train, optimize_test, final_train, final_test


def load_pickled(name):
    # load_path = path + "\\" + name
    load_path = "DATA/" + name
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
        return data


def load(name):
    return assign(load_pickled(name), name)


def normalized(tensor):
    print("var:")
    for i in range(tensor.size(2)):
        print(torch.var(tensor[:, :, i]))
    print("mean:")
    for i in range(tensor.size(2)):
        # print("dim " + i.__str__())
        print(torch.mean(tensor[:, :, i]))


def denormalizer(data, name):
    norm_data = load_pickled(name)
    normalizer = norm_data.get('normalizer')
    norm_observations = normalizer.get('observations')
    out = denormalize_(data, norm_observations)
    return out


def print_data(data_dict):
    print("Train Obs Shape", data_dict['train_obs'].shape)
    print("Train Act Shape", data_dict['train_act'].shape)
    print("Train Targets Shape", data_dict['train_targets'].shape)
    print("Test Obs Shape", data_dict['test_obs'].shape)
    print("Test Act Shape", data_dict['test_act'].shape)
    print("Test Targets Shape", data_dict['test_targets'].shape)
    print("Normalizer", data_dict['normalizer'])
