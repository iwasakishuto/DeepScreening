## data

|file name|description|available from|
|`250k_rndm_zinc_drugs_clean_3.csv`|Contains SMILES, logP, QED, SAS for 250 chemicals.|[aspuru-guzik-group/chemical_vae](https://github.com/aspuru-guzik-group/chemical_vae/blob/a509dc613c4c9be54d448a34829bab25bcf09a79/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv)|
|`Gene_co-expression_modules.xlsx`|Supplementary Table S3 Gene co-expression modules and member genes.|[Characterization of cancer omics and drug perturbations in panels of lung cancer cells](https://doi.org/10.1038/s41598-019-55692-9)|

### train/test data

All train and test subsets are divided by the following method:

```python
from kerasy.utils import train_test_split
(x_train,x_test),(y_train,y_test) = train_test_split(x,y, test_size=0.2, random_state=0)
```

|train/test_data||
|`smiles.npy`|`smiles` column of `250k_rndm_zinc_drugs_clean_3.csv`.|
|`chemvae_reg.npy`|`logP` and `SAS` columns of `250k_rndm_zinc_drugs_clean_3.csv`.|
|`chemvae_logit.npy`|`qed` column of `250k_rndm_zinc_drugs_clean_3.csv`.|


### Specific example

Look at the notebook: [Preprocessing.ipynb](https://nbviewer.jupyter.org/github/iwasakishuto/DeepScreening/blob/master/notebook/Preprocessing.ipynb)
