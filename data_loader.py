import pickle
from unipath import Path


path = Path('E:\\uni kram\\BA_NEW\\DATA')


def load_pickled(name):
    # load_path = path + "\\" + name
    load_path = "DATA\\" + name
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
        return data
