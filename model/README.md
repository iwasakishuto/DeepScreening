## ChemVAE

sample model for `ChemVAE`.

```sh
$ python deepscreening/chemvae.py -p model/chemvae/params.yml
```

### How to create data??

```python
import json
import numpy as np
import pandas as pd
from deepscreening.utils import smiles2onehot
from kerasy.utils import train_test_split

df = pd.read_csv("data/250k_rndm_zinc_drugs_clean_3.csv")
smiles  = df.smiles.values
y_reg   = df[["logP", "SAS"]].values
y_logit = df.qed.values
X, idx2smiles = smiles2onehot(smiles, return_meta=True)

with open("model/chemvae/idx2smiles.json", mode='w') as f:
    json.dump(idx2smiles, f)

(x_train,x_test),(y_reg_train, y_reg_test),(y_logit_train, y_logit_test) = train_test_split(X, y_reg, y_logit, random_state=0, test_size=0.3)

np.save("model/chemvae/train_x.npy", x_train)
np.save("model/chemvae/train_y_reg.npy", y_reg_train)
np.save("model/chemvae/train_y_logit.npy", y_logit_train)
np.save("model/chemvae/test_x.npy", x_test)
np.save("model/chemvae/test_y_reg.npy", y_reg_test)
np.save("model/chemvae/test_y_logit.npy", y_logit_test)
```
