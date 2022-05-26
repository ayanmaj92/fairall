# General Format For Datasets
*Please ensure* the following points are satisfied if you wish to include new data.
## a. Data directory
Create a new directory for your new dataset in the `datasets/` folder, e.g., `mkdir datasets/{NEW_DATASET}`.
## b. Data files
Ensure the following files are in the newly created directory (*with the same names as shown here*):
- Files for pre-training (phase 1) and/or warmup (before phase 2):
  - training data: `data1.csv`
  - training data variable types: `data1_types.csv`
  - validation data: `valid1.csv`
  - validation data variable types: `valid1_types.csv`
  - test data: `test1.csv`
  - test data variable types: `test1_types.csv`
- Files for decision-making (phase 2):
  - training data: `data2.csv`
  - training data variable types: `data2_types.csv`
  - validation data: `valid2.csv`
  - validation data variable types: `valid2_types.csv`
  - test data: `test2.csv`
  - test data variable types: `test2_types.csv`
- (Optional for synthetic data) Counterfactual test data:
  - counterfactual test data: `cf_test2.csv`

For details about how to generate counterfactuals, refer to the paper appendix.
## c. CSV file format
The columns of the CSV `data` files and the `types` files need to follow certain formats.

1. Ensure that the `data1.csv` file has the following columns (*in order*):

```{Features columns X_1, X_2, ...}, {Utility Label Y}, {Sensitive S}, {Policy_1 (probability of decision p(D=1|X,S))}, {Policy_1_D (exact decision)}, {Policy_2...}, ...```

For instance for `datasets/compas/data1.csv` we have the following:

```age_cat,priors_count,c_charge_degree,Y,S,LENI,LENI_D,HARSH,HARSH_D```

So here, `age_cat,priors_count,c_charge_degree` are the data features `X`.
Note here we have two policies, `HARSH` and `LENI`. So, the columns `LENI` and `HARSH` contain the probabilities `P(D=1|X,S)`, 
whereas the columns `LENI_D` and `HARSH_D` contain the exact decision.

2. `valid1.csv` and `test1.csv` should have all columns *up to* `Y,S`. So, these files do not need to have the policy-related columns.

For instance for `datasets/compas/valid1.csv` we have the following:

```age_cat,priors_count,c_charge_degree,Y,S```

3. Note, synthetic datasets may have additional column corresponding to the `ground-truth` utility label. This column would be `Y_fair`. This column would be placed as `...,Y,S,Y_fair,{Policy},...` in the CSV files.
4. All phase 2 files (`data2.csv, valid2.csv, test2.csv, (optionally) cf_test2.csv`) should have all columns *up to* `Y,S`.

For instance for `datasets/compas/` there are these columns only:

```age_cat,priors_count,c_charge_degree,Y,S```

5. The feature values should also follow certain encoding:
  - Sensitive feature `S` is assumed to be binary, and should be encoded as `{1, -1}` for the `{advantaged, disadvantaged}` groups respectively.
  - Label `Y` and (optionally) `Y_fair` are assumed to be binary, and should be encoded as `{1, 0}`.
  - Policy decisions, e.g., `HARSH_D, LENI_D` should also be binary `{1, 0}`.
  - Any binary feature `X` should be encoded as `{1, 0}`.
  - Any categorical feature `X` should be **label-encoded**. E.g., in `datasets/compas/` the feature `age_cat` can have 3 categories. So, it is encoded as `{0, 1, 2}`. Hence, each data-point can have `age_cat` as one of the 3 values.
6. For each of the CSV files (`data, valid, test` for phase 1 and 2) should also have their corresponding `{data}{phase}_types.csv` file. This file should contain the specific data-types for each feature and which columns they occupy in the files.

### Data types file

For instance, in `datasets/compas/`, for phase 1 CSV files, `data1_types.csv` (or `valid1_types.csv`, `test1_types.csv`) is:
```text
type,dim,nclass
cat,3,3
count,1,
cat,2,2
cat,2,2
```
Note that correspondingly: 
- `age_cat` is categorical, 3 dimensional with 3 classes
- `priors_count` is count variable with 1 dimension
- `c_charge_degree` is categorical, 2 dimensional with 2 classes
- `S` is likewise as above

But for phase 2 CSV files, `data1_types.csv` (or `valid1_types.csv`, `test1_types.csv`) is:
```text
type,dim,nclass
cat,3,3
count,1,
cat,2,2
cat,2,2
cat,2,2
```

Note that now there is an **additional** `cat,2,2` in the last line. This additional information is to handle the utility (label) variable in the probabilistic model.

*It is important to note* that this additional `cat,2,2` should be there for all datasets!

The supported data-types are:
- `real`
- `count`
- `cat` (categorical, includes binary as well)