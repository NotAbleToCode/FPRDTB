# Fast Provably Robust Decision Trees and Boosting
The implementation of Fast Provably Robust Decision Trees and Boosting.

## Simple example
### FPRDT
To train and evaluate FPRDT on dataset mnist1v7, one can use the following command:
```
python main.py -m FPRDT --dataset mnist1v7
```

### PRAdaBoost
To train and evaluate PRAdaBoost on dataset mnist1v7, one can use the following command:
```
python main.py -m PRAdaBoost --dataset mnist1v7
```

### Other optional parameters
```python
'''
--max_tree_num: int
  The number of maximum trees in PRAdaBoost.
--max_depth: int 
  The maximum depth of the FPRDT.
--min_samples_split: int 
  The minimum samples needed to split a leaf.
--min_samples_leaf: int
  The minimum samples needed to create a leaf.
--epsilon: float
  The maximum perturbation size.
--k_times: int
  The times to evaluate the model.
--k_folds: int
  The number of folds in each evaluating time.
--random_seed: int
  The random seed used to split the dataset.
--record_loss
  If record the curve of the loss in the training process.
--record_img
  If generate adversarial images after training process.
'''
```
