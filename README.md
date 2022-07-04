# FairAll: Utilizing Unlabeled Data for Fair Decision Making
<a href="https://arxiv.org/abs/2205.04790"><img src="https://img.shields.io/badge/preprint-arXiv%3A2205.04790-darkred" class="center" target="_blank" rel="noopener noreferrer"></a>
<br><br>
![image saying no to man littering](./no-litter.png)

This is the code repository for the ACM FAccT 2022 paper **"Don't Throw it Away! The Utility of Unlabeled Data in Fair Decision Making"** ([arXiv](https://arxiv.org/abs/2205.04790)). 
The implementation is based on [Pytorch](https://pytorch.org/) and 
 [Ignite](https://pytorch.org/ignite/v0.3.0/index.html). 
The repository contains the necessary resources to train the models and collect the metrics for 
evaluating the experiments in the paper, including the datasets.

## Setup conda environment
We assume that [conda](https://www.anaconda.com/) is already installed. For the ease of setup, we provide a 
`yml` environment setup file. 
#### a. Create conda environment
Give the following command to create the environment. 
```
conda env create -f fairall_env.yml
```
This creates a conda environment named `fairall`.
#### b. Activate conda environment
To activate the created environment, use the following command.
```
conda activate fairall
```

## Datasets

The repository provides all the datasets that have been used for the experiments in the paper. These include:
- Synthetic dataset (`datasets/SCB_11/`)
- [COMPAS](https://github.com/propublica/compas-analysis) (`datasets/compas/`)
- [German Credit](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29) (`datasets/credit/`)
- [MEPS Health](https://meps.ahrq.gov/mepsweb/) (`datasets/meps/`)

The latter 3 datasets are real-world datasets that have been downloaded using 
[AIF360](https://github.com/Trusted-AI/AIF360). Please see the related licences. We also refer to the [user agreement for MEPS](https://github.com/Trusted-AI/AIF360/blob/master/aif360/data/raw/meps/README.md#data-use-agreement). 
We have pre-processed the datasets for our compatibility and as described in our paper. 

Read [here](./datasets/README.md) for details on the key points regarding 
*dataset formatting for compatibility.*

## How to run the code
The standard running format is:
```shell
python main.py --dataset_file {DATASET_PARAMS_FILE} --model_file {MODEL_PARAMS_FILE} --trainer_file {TRAINER_PARAMS_FILE}
```
Note that all params `yaml` files can be found in `_params/` folder. New params files should also be created here, if necessary.

Optionally, you can give `-r` for results root directory, and `-s` for specific seed. 

E.g., one instance to run FairAll (I+II) for synthetic data would be: 

```shell
python main.py --dataset_file _params/dataset_SCB_11.yaml --model_file _params/model_fairall.yaml --trainer_file _params/trainer.yaml -r results/exper_SCB_11_fairall_12 -s 100
```

Note that just giving `python main.py` would just start executing FairAll (I+II) on synthetic data with some default parameter files provided.

You can check all the flags with:
```shell
python main.py --help
```
For the sake of completeness, here is the output of the argument `--help`:
```buildoutcfg
usage: main.py [-h] [--dataset_file DATASET_FILE] [--model_file MODEL_FILE]
               [--trainer_file TRAINER_FILE] [-d KEY1=VAL1,KEY2=VAL2...]
               [-m KEY1=VAL1,KEY2=VAL2...] [-o KEY1=VAL1,KEY2=VAL2...]
               [-t KEY1=VAL1,KEY2=VAL2...] [-s SEED] [-r RESULT_DIR]

Authors: Ayan Majumdar, Miriam Rateike

optional arguments:
  -h, --help            show this help message and exit
  --dataset_file DATASET_FILE
                        path to configuration file for the dataset
  --model_file MODEL_FILE
                        path to configuration file for the dataset
  --trainer_file TRAINER_FILE
                        path to configuration file for the training
  -d KEY1=VAL1,KEY2=VAL2..., --dataset_dict KEY1=VAL1,KEY2=VAL2...
                        manually define dataset configurations as string:
                        KEY1=VALUE1+KEY2=VALUE2+...
  -m KEY1=VAL1,KEY2=VAL2..., --model_dict KEY1=VAL1,KEY2=VAL2...
                        manually define model configurations as string:
                        KEY1=VALUE1+KEY2=VALUE2+...
  -o KEY1=VAL1,KEY2=VAL2..., --optim_dict KEY1=VAL1,KEY2=VAL2...
                        manually define optimizer configurations as string:
                        KEY1=VALUE1+KEY2=VALUE2+...
  -t KEY1=VAL1,KEY2=VAL2..., --trainer_dict KEY1=VAL1,KEY2=VAL2...
                        manually define trainer configurations as string:
                        KEY1=VALUE1+KEY2=VALUE2+...
  -s SEED, --seed SEED  set random seed
  -r RESULT_DIR, --result_dir RESULT_DIR
                        directory for storing results
```
### Running on different available datasets
Each dataset has it's corresponding param `yaml` file in `_params/`, e.g., `_params/dataset_SCB_11.yaml`, `_params/dataset_compas.yaml`, `_params/dataset_credit.yaml`, `_params/dataset_meps.yaml`.
To run on any of these datasets, simply pass the param file location with the `--dataset_file` flag.

### Changing some of the parameters for training
How do we run the code by changing some of the parameter?
* One option would be to change the parameter `yaml` files. Read `Parameter Files` below. Editing these files and running is one option.
* Another option would be to pass changed parameters as arguments (using `-m`, `-t`, `-d` options). 
 
One example of specifically using the flags to change some parameters could be:
```shell
python main.py --dataset_file _params/dataset_SCB_11.yaml --model_file _params/model_fairall.yaml --trainer_file _params/trainer.yaml -r results/exper_SCB_11_fairall_12 -m learn_policy=FZ+fz_method=CLF+costs=0.2 -d init_policy=RAN+percent=0.7
```

Here, we used:
1. `-m learn_policy=FZ+fz_method=CLF+costs=0.2` to change the model parameters for policy, method and cost of decision.
2. `-d init_policy=RAN+percent=0.7` to change initial policy to random and change the acceptance percentage for the random policy.

Note, how different parameters (for model and dataset) are separated with `+`.
Read `Parameter Files` below for details about the different parameters available.

### Examples of running different policy methods
We give some examples for running each of the different policy methods we report.
Note, for these examples we use synthetic dataset (`_params/dataset_SCB_11.yaml`) and the provided `_params/trainer.yaml`.

#### VAE Phase I
Taking the optimal values from `Table 6` from the `Appendix` of the paper for `Synthetic` data:
```shell
python main.py --dataset_file _params/dataset_SCB_11.yaml --model_file _params/model_fairall.yaml --trainer_file _params/trainer_phase1.yaml -r results -s 100 -o learning_rate=0.005 -m beta=0.8
```
Note, for optimal results, the saved checkpoint and `hparams yaml` files should be loaded directly for FairAll (I+II) learning. See below `Loading phase 1 checkpoint directly` for more details.

#### FairAll (I+II)
For training phases 1 and 2 together, we use with FZ-DEC policy.

```shell
python main.py --dataset_file _params/dataset_SCB_11.yaml --model_file _params/model_fairall.yaml --trainer_file _params/trainer.yaml -r results -s 100
```

#### FairAll (II)
We simply need phase 1 training off.

```shell
python main.py --dataset_file _params/dataset_SCB_11.yaml --model_file _params/model_fairall.yaml --trainer_file _params/trainer.yaml -r results -s 100 -t training=False
```

#### FairLab (I+II)
We need different loss, both phase training and FZ-LAB policy.

```shell
python main.py --dataset_file _params/dataset_SCB_11.yaml --model_file _params/model_fairlab.yaml --trainer_file _params/trainer.yaml -r results -s 100
```

#### FairLog
Competing method based off Kilbertus et al. "Fair Decisions Despite Imperfect Predicitions". No phase 1 training, different loss function and policy.

```shell
python main.py --dataset_file _params/dataset_SCB_11.yaml --model_file _params/model_fairlog.yaml --trainer_file _params/trainer.yaml -r results -s 100 -t training=False
```

#### UnfairLog
The unfair version of the competing method FairLog. Set lambda to 0 to have no fairness constraint and only optimize utility.

```shell
python main.py --dataset_file _params/dataset_SCB_11.yaml --model_file _params/model_fairlog.yaml --trainer_file _params/trainer.yaml -r results -s 100 -t training=False -m lambda=0
```

**Note,** depending on dataset and policy method, other hyperparameters (like model architecture, etc.) might need to be changed. Refer to `Appendix` of paper for details.
These parameters can be accessed and modified. For more details about the different parameters, read below.

## Parameter Files
Running the code first requires providing parameters for the 3 different components of our pipeline: dataset, model and the trainer.
These should be configured in the `_params/` folder.
### Dataset parameters file
Each dataset should have its own parameter file in `_params/{DATASET}.yaml`. This file should have a particular format. 
For instance, for COMPAS dataset, `_params/dataset_compas.yaml` is as follows:
```yaml
dataset:
  name: compas
  params1:
    std: True # Standardize data
  params2:
    batch_size: 128
    init_policy: HARSH
    percent: 0.3 # percentage labeled datapoints, if initial policy RANdom
```
* `name` should match the folder name for the dataset.
* `std: True` indicates standardize the features of the data before learning.
* `batch_size` indicates the batch size to use for training in phase 1.
* `init_policy` is the &pi;<sub>0</sub> starting policy used in warmup. Our work uses `HARSH`, `LENI`. Can also be `RAN` (random).
* `percent` is only used for random `RAN` policy to indicate the fraction of people to accept.

**Any parameter in `params2` can be changed using the arguments of `main.py`.** Give `-d` option and separate each with `+`, e.g.:

```-d init_policy=RAN+percent=0.7```

### Model parameters file
We provide parameter files for different models. The parameter files again follow a certain format. 
An example would be `_params/model_fairall.yaml`:
```yaml
optimizer:
  name: adam
  params:
    learning_rate: 0.001

model:
  name: sscvae
  params:
    L: 50 #samples of Z to evaluate ELBO
    K: 100 #samples for MC estimation of unsupervised KL
    latent_size: 2 #latent dimension of z
    h_dim_list_clf: [64, 64]
    h_dim_list_enc: [64, 64]
    h_dim_list_dec: [64, 64]
    act_name: relu #activation function
    drop_rate_clf: 0.1 #dropout rate
    conditional: True #conditional VAE
    alpha: 10 #factor for classifier loss
    lambda: #NA
    beta: 0.8 #annealing factor for KL
    costs: 0.5 #costs of positive decision
    learn_policy: FZ #'QXS', 'PZS', 'FZ', 'NA'
    fz_method: DEC #'CLF', 'DEC', 'LAB' (IPS)
    pol_sampler: LOG  # One of DET, LOG
    loss_function: loss_fairall
    model_type:  #leave empty for None, only relevant for niki (fair/unfairlog)
    phase1-ckpt: #leave empty for None or paste here model with the entire model path
```

The components are (note that components that are unused by specific models can be left empty):
* `optimizer` with the name (we use `adam`) and with `params` as `learning_rate` (we use `0.001`).
* `model` that defines the core model components:
  * `name`: this is always `sscvae` as this is the backbone model object we use.
  * `L,K`: samples for evaluating ELBO and KL respectively. Not needed to change.
  * `latent_size`: VAE latent size. Change according to data.
  * `h_dim_list_enc,h_dim_list_dec`: hidden layer sizes as list for encoder and decoder. Used for `fairall` and `fairlab`.
  * `h_dim_list_clf`: hidden layer sizes as list for classifier. Used by `fairall`, `fairlog (unfairlog)`.
  * `act_name`: activation used, we fix this to `relu`.
  * `drop_rate_clf`: the dropout rate `p` for `torch Dropout`. Only for classifier model.
  * `conditional`: set to `True` for `fairall` and `fairlab`. Not used by `fairlog (unfairlog)`.
  * `alpha`: the weight to classifier loss. Used by `fairall`.
  * `lambda`: fairness constraint weight used for FairLog and UnfairLog. Set `lambda: 0` for UnfairLog and `lambda: {positive number}` for FairLog.
  * `beta`: KL weight applied to ELBO loss. Used by `fairall` and `fairlab`.
  * `costs`: cost of positive decision in the policies and classifier. Depends on the dataset.
  * `learn_policy`: the policy to use for learning. Can be `Classifier: QXS (fairall, fairlog, unfairlog)`, `Decoder: PZS (fairall, fairlab)` and `Latent Z: FZ (fairall, fairlab)`.
    * For `fairlog (unfairlog)` we use `QXS`. 
    * For `fairall` and `fairlab` we use `FZ`.
  * `fz_method`: (only `fairall` and `fairlab`) if using latent Z for policy. Can be `CLF (label with classifier)`, `DEC (label with decoder)` and `LAB (use only labeled data)`.
    * For `fairall` we use `DEC`.
    * For `fairlab` it is **always** `LAB`.
  * `pol_sampler`: the policy decision sampling method. Can be stochastic `LOG` or deterministic `DET`. We use `LOG` in our paper.
  * `loss_function`: the loss function to use. Can be for FairAll `loss_fairall`, for FairLab `loss_fairlab` and for FairLog and UnfairLog `loss_fairlog`. 
    * Note that to differentiate FairLog and UnfairLog, we use the `lambda` parameter.
  * `model_type`: relevant for FairLog and UnfairLog. Can be `nn` or `lr` for neural network or logistic regression. We fix it to `nn`.
  * `phase1-ckpt`: only for `fairall`. If we want to directly load a `ckpt` file, give the **full path** to the `ckpt` file here. 

Note that a lot of these parameter values would be set dependent on the dataset being used. Check the appendix for details.

**Any parameter in `optim` can be changed using the arguments of `main.py`.** Give `-o` option. Can only change `learning_rate`.

**Any parameter in `model` can be changed using the arguments of `main.py`.** Give `-m` option and separate different ones with `+`, e.g.:

`-m learn_policy=FZ+fz_method=CLF+costs=0.2`

### Trainer parameters file
Finally, the training parameters are provided in the `trainer.yaml` file. We provide an example here for understanding the format.
```yaml
root_dir: results # Root folder to save the model
trainer:
  print_every: 1
trainer1:
  training: True #If True, train phase 1
  epochs: 1200 # Maximum number of epochs to train
  phase1_samples: #leave like this, we only used this for cross-validation
trainer2:
  training: True #If True, train phase 2
  only_warmup: False #If True, only do warmup phase (this is True only for cross validation)
  epochs_per_ts: 1 # Number of epochs per time step
  time_steps: 200 # Number of time steps
  samples_per_ts: 64 # Number of samples per time step
  warmup_ts: 50  # How many epochs to do warmup?
  warmup_samples: 128 # How many samples do we consider as applicants for warmup?
```
The following points are **important** to note:
* For FairAll (I+II) set `trainer1`'s `training: True`.
* For FairAll (II) set `trainer1`'s `training: False`.
* For FairLab (I+II) set `trainer1`'s `training: True`.
* For FairLog and UnfairLog set `trainer1`'s `training: False`.
* For general training, **always** set `trainer2`'s `only_warmup: False`.
* `root_dir`: set the root directory for storing results, usually it would be, e.g., `results/`.
## Output results
All results are output to a folder. This location is printed out to console at the end of training:

`All results saved to: {save_directory}`.

So, `cd` to this directory. This is where all output files can be found.

### JSON files
All crucial results, especially those reported in the paper are dumped to JSON files. 
For FairAll (I+II), FairLab (I+II) we would have 6 JSON files, 3 for each phase (each reporting for training, validation and test data).
For FairAll (II), FairLog, UnfairLog, we would have 3 JSON files (only for phase 2).

The files are named as `output{phase}{data}.json`. 
`{phase}` can be `1,w,2` for phase 1, warmup or 2 respectively.
`{data}` can be `train,valid,test` respectively.

### JSON Metrics Of Interest
The following metrics can be found in the JSON files under `metrics` key. Note that not all metrics will be present all the time.
The presence of a metric depends on which model (and loss) we use and which data we use for training.
- Data feature reconstruction: Real-valued features `Rec_NRMSE_X{i}`, Categorical features `Rec_ERR_X{i}`. These are only for FairAll, FairLab.
- Utility reconstruction of decoder: `Rec_ERR_U` (only for FairAll, FairLab)
- Classifier error for utility: `Clf_ERR_U` (only for FairAll, FairLog, UnfairLog)
- Policy related: utility (`Policy_utility`), error (`Policy_ERR`)
- Fairness related: demographic parity (`DP_unfairness`), counterfactual fairness (`CF_unfairness`, only for synthetic data)
- Training effectiveness: accumulated utility `Effective_util`, accumulated DP unfairness `Effective_DPU` (only found in `output{phase}train.json`)

**Additional measures for ground truth**: For synthetic data, if we also have ground truth available, 
we additionally collect `Policy_Gnd_utility`, `Policy_Gnd_ERR`, `Effective_util_gnd`. 


### Other outputs
The other *outputs of interest* generated are as follows:
- `yaml` files for params for training each phase as `hparams_phase{phase}.yaml` (1, 2, warmup)
- 2-dimensional PCA images visualizing latent Z as `latent_phase_{phase}.png` (1, 2, w)
- Images for original and reconstructed marginal feature distributions as `marginal_phase_{phase}_{feature_dim}.png` where phase (1, 2, w) and feature_dim (0, 1, ...)
- Prediction error of `S` from `Z` as `pred-s-from-z-err-ph-{phase}.txt` (1, 2, w)
- Prediction error of `U` from `Z` as `pred-u-from-z-err-ph-{phase}.txt` (1, 2, w)
- Training times stored in `time.json`
- Phase 1 checkpoint and Phase 2 checkpoint(s) as `.pth` files. Note to always use the latest checkpoint files (order by time).

## Loading phase 1 checkpoint directly
One can load a pre-existing phase 1 model checkpoint for running subsequent phase 2 operations.
One example situation could be running the code with only phase 1 training turned on.
Use `_params/trainer_phase1.yaml` for running only phase 1 (change any parameters as desired, e.g., `trainer1:epochs`).

For loading phase 1 checkpoint, give the location of the `.pth` checkpoint file and the phase 1 `hparams yaml` file.
Note these files would be inside the corresponding results directory as `checkpoint_phase1_model_total_loss={...}.pth`
and `hparams_phase1.yaml`.

If desired, copy these files to other locations. 
To load the checkpoint related files, there are 2 options.
1. Edit `model_fairall.yaml` or `model_fairlab.yaml`. Put the path locations of the `.pth` in `model: params: phase1-ckpt`.
Also put the location of the `hparams_phase1.yaml` in `model: params: phase1-hparams`.
2. Directly put this information in the console while executing code:

```shell
python main.py --dataset_file {DATASET PARAMS YAML} --model_file {MODEL PARAMS YAML} --trainer_file {TRAINER PARAMS YAML} -m phase1-ckpt={Location of checkpoint_phase1 pth file}+phase1-hparams={Location of hparams_phase1.yaml file}
```

## Contact
For any queries, please contact `ayanm{at}mpi-sws.org` and `mrateike{at}tue.mpg.de`.

## Cite Us
To cite this work, please cite the main paper
```text
@inproceedings{rateike2022don,
  title={Don’t Throw it Away! The Utility of Unlabeled Data in Fair Decision Making},
  author={Rateike, Miriam and Majumdar, Ayan and Mineeva, Olga and Gummadi, Krishna P and Valera, Isabel},
  booktitle={2022 ACM Conference on Fairness, Accountability, and Transparency},
  pages={1421--1433},
  year={2022}
}
```

## Acknowledgement
The code in this repository for training VAE models on heterogeneous data is based on an older version of [this repo](https://github.com/adrianjav/heterogeneous_vaes).
Thanks to [Adrián Javaloy Bornás](https://github.com/adrianjav/) for sharing his code!
