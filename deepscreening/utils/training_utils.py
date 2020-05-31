# coding: utf-8
import os
import numpy as np
import pandas as pd
from kerasy.utils import CategoricalEncoder
from kerasy.utils import toCYAN

from .generic_utils import pad_string

def smiles2onehot(smiles, chars=None, max_len=None, return_meta=False):
    """
    @params smiles: (list) SMILES strings.
                           ex.)
                           'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1',
                           'C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1',
                           'N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1',
                           ....
    @params chars   : (str)
    @params max_len : (int)
    """
    # Get rid of common mistakes.
    smiles = [smile.rstrip("\n") for smile in smiles]

    # Padding the smiles.
    if max_len is None:
        max_len = max([len(smile) for smile in smiles])
    smiles = [pad_string(e, max_len, padding="right") for e in smiles]

    # one-hot encoding.
    encoder = CategoricalEncoder()
    encoder._mk_obj2cls_dict([e for smile in smiles for e in smile])
    num_classes = len(chars) if chars is not None else None
    one_hot = np.asarray([encoder.to_onehot(smile, num_classes=num_classes) for smile in smiles], dtype=int)
    if return_meta:
        return (one_hot, encoder.cls2obj)
    else:
        return one_hot

def load_data(path, **kwargs):
    """ Load data according to the extension. """
    if path is None:
        data = None
    else:
        fn, ext = os.path.splitext(path)
        if ext == ".npy":
            data = np.load(path, **kwargs)
        elif ext in [".csv", ".txt"]:
            data = pd.read_csv(path).values()
        elif ext in [".xls", ".xlsx"]:
            data = pd.read_excel(path).values()
    return data

def arange_datasets(x_data={}, y_data={}, datasets="", basedir=None, **kwargs):
    """
    NOTE: File path is `basedir/x(y)_{datasets}_{fn}`
    ~~~
    @params x_data   : (dict) {"input_layer"  : filename}
    @params y_data   : (dict) {"output_layer" : filename}
    @params basedir  : (str)  path/to/data/directory/
    @params datasets : (str)  prefix of the filename.
    @params kwargs   : kwargs for `load_data`.
    """
    if dir is None:
        return None
    else:
        x_data_dict = {layer: load_data(os.path.join(basedir,f"x_{datasets}_{fn}"), **kwargs) for layer,fn in x_data.items()}
        y_data_dict = {layer: load_data(os.path.join(basedir,f"y_{datasets}_{fn}"), **kwargs) for layer,fn in y_data.items()}
        return (x_data_dict, y_data_dict)

def arange_all_datasets(params, **kwargs):
    train_dir = params.get("train_dir")
    val_dir   = params.get("val_dir")
    test_dir  = params.get("test_dir")
    x_data    = params.get("x_data")
    y_data    = params.get("y_data")

    x_train_data, y_train_data = arange_datasets(x_data, y_data, datasets="train", basedir=train_dir, **kwargs)
    x_val_data, y_val_data     = arange_datasets(x_data, y_data, datasets="val",   basedir=val_dir, **kwargs)
    x_test_data, y_test_data   = arange_datasets(x_data, y_data, datasets="test",  basedir=test_dir, **kwargs)

    return (x_train_data,y_train_data),(x_val_data,y_val_data),(x_test_data,y_test_data)
