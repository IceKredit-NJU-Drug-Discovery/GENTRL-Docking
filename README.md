# GENTRL-Docking

[![license](https://img.shields.io/github/license/microsoft/molecule-generation.svg)](https://github.com/microsoft/molecule-generation/blob/main/LICENSE)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)

This repository contains a modified version of In Silico Medicine's GENTRL algorithm trained on approximate docking scores, 
introduced in [Improving Drug Discovery with a Hybrid Deep Generative Model Using Reinforcement Learning Trained on Bayesian Feature Representation]

## install

`gentrl-docking` additionally depends on `rdkit` and appropriatey configured CUDA libraries.
Our package was tested with `python=3.7`, `pytorch=1.10.2` and `rdkit=2020.09.1`; 
see the `environment.yml` files for the exact configurations.

## Workflow

Working with GENTRL can be roughly divided into four stages:
*data preprocessing*, users should provide a csv file list of SMILES strings containing descriptions of the molecular properties,
*training*, where GENTRL is trained on the preprocessed data until convergence;
*sampling*, where new molecules are generated from the trained model;
*reinforcement learning*, where a previously trained model is fine-tuned based on sampled molecules and reward function.

### Data Preprocessing

To run preprocessing, your data has to follow a simple csv format, each containing SMILES strings and properties.
a sample pretraining dataset can be found here: https://drive.google.com/file/d/1N5Rny0_JnFVAkWJpHfC1Nzs-MgU5eLH2/view?usp=sharing

Then, the model can load the dataset in format:
```
 {'path':'ddr1_datasets/ZINC_IHAD_~100k_clean.csv',
         'smiles': 'smiles',
         'prob': 0.175,
         'label' : 'label',
 }
 `path` is the dataset location;
 `smiles` is the columns name for SMILES strings;
 `label` is the columns name for label;
 `prob` is the proportion of this dataset in the whole training data;
```

### Training

Having stored some preprocessed data under `ddr1_datasets`, the model can be trained by running codes on `Pretrained_Generation_Example.ipynb`

After running codes, you should get an output model file as:
```
    dec.model
    enc.model
    lp.model
    order.pkl
```
A sample model is shown in /model/Bayes_model_iter599


### Sampling

`sample.ipynb`  could generate new molecules in the form of SMILES strings. There is a existed reinforcement learning model under `/model/Bayes_model_iter599`  which could be used.  

This code could be used to load the model:

```python
model.load('../model/Bayes_model_iter599')
```

Following code could help to sample, remove the invalid molecules from the sample and merge the effective number of duplicates.

```
def gen_mol(model, n_r):
    generated = []
    for i in range(n_r):
        sampled = model.sample(100)
        sampled_valid = [s for s in sampled if get_mol(s)]
        generated += sampled_valid
    df = pd.DataFrame(generated, columns = ['SMILES'])
    res_count = pd.DataFrame(df.value_counts()).rename({0:'count'},axis = 1).reset_index()
    return res_count

`model` the model to be sampled;
`n_r` the amount of sampling SMILES including invalid and duplicated results is 100 times of n_r.
```



### Reinforcement Learning

Having pretrained a model from previous steps, the RL step can be implemented by running codes on `Reinforcement_Learning_Example.ipynb` 

the sample reward function looks like:
```
    def fn(smiles,reward_fn = bayes_regression,fn):
        mol = get_mol(mol_or_smiles)
        
	xx = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol_or_smiles), 2, nBits=2048)
        xx = np.vstack(xx).reshape(1,-1)
        
	mfp_sum = np.array(xx).sum()
        mfp = 1 / (1+np.exp(-(mfp_sum-60)/10))
        
	bayes_regression = np.exp(-bayes_regression.predict(np.array(xx).reshape(1,-1))[0])
        reward = mfp * bayes_regression
        
	return reward
```
`smiles` the SMILES strings sampled from pretrained model;
`reward_fn` the applied reward function

The code first samples molecules based on pretrained model, and then the  RL process will maximize the value of reward funciton to achieve a fine-tuned model.

Feel free to contact us if you have any suggestions.

## Authors
* [Chris Butch](mailto:chrisbutch@nju.edu.cn)
* [Youjin Xiong](mailto:xiongyoujin@foxmail.com)
* [Linyun Gu](mailto:gu_lingyun@icekredit.com)
* [Yiqing Wang](mailto:yiqingwangusc@gmail.com)
* [Junyu Wu](mailto:wu_junyu@icekredit.com)
