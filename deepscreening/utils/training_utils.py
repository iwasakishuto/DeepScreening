# coding: utf-8
import numpy as np
from kerasy.utils import CategoricalEncoder
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
