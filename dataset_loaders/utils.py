"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torchvision.datasets.folder import default_loader
import pickle

def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except BaseException:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None

    return img


def load_pickle(pickle_file):
    try:
        pickle_data = pickle.load(pickle_file)
    except UnicodeDecodeError as e:
        pickle_data = pickle.load(pickle_file, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def open_load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data