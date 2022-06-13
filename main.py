import sys 
import getopt
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from sklearn import tree
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from fprdt import FPRDecisionTree
from pradaboost import PRAdaBoost
from datasets import load_all, load_dataset, load_epsilons_dict
from util import *

def getpars():
    """
    Obtain all super parameters.
    """
    opts, args = getopt.getopt(sys.argv[1:],'m:n:',\
        ['max_depth=', 'min_samples_split=', 'min_samples_leaf=', \
            'max_features=', 'dataset=', 'epsilon=', 'k_times=', \
            'k_folds=', 'record_loss', 'record_img', 'if_sample', \
            'random_seed=', 'max_tree_num=', 'max_attempt_num=', 'all'])
    tmp = opts
    opts = {}
    for (k, v) in tmp:
        opts[k] = v
    if not ('-m' in opts):
        opts['-m'] = 'FPRDT'
    if not ('-n' in opts):
        opts['-n'] = 100
    if not ('--max_depth' in opts):
        opts['--max_depth'] = '999999'
    if not ('--min_samples_split' in opts):
        if opts['-m'] == 'FPRDT':
            opts['--min_samples_split'] = '10'
        else:
            opts['--min_samples_split'] = '10'
    if not ('--min_samples_leaf' in opts):
        if opts['-m'] == 'FPRDT':
            opts['--min_samples_leaf'] = '5'
        else:
            opts['--min_samples_leaf'] = '5'
    if not ('--max_features' in opts):
        opts['--max_features'] = None
    if not ('--k_times' in opts):
        opts['--k_times'] = '5'
    if not ('--k_folds' in opts):
        opts['--k_folds'] = '5'
    if not ('--random_seed' in opts):
        opts['--random_seed'] = None
    if not ('--max_tree_num' in opts):
        opts['--max_tree_num'] = '100'
    if not ('--max_attempt_num' in opts):
        opts['--max_attempt_num'] = '30'
    return opts

def train_kfold_model(opts):
    """
    Train and test model on (X,y) k_times times. Everytime k_folds folds will be done on the datasets.

    Parameters
    ----------
    model: object
    X: array-like of shape (n_samples, n_features)
    y: array-like of shape (n_samples, )
    k_times: int
        Repeat training and testing k_times times.
    k_folds: int
        Everytime a k_folds cross validation will be run.
    record_loss: bool
        Whether to record the loss during the training process.
    record_img: bool
        Whether to record the imgs and the corresponding adversarial imgs after training.

    Returns
    --------- 
    depths: list[float]
        A list with k_times*k_folds size to record the depths. 
    leaf_nums: list[float]
        A list with k_times*k_folds size to record the leaf_nums. 
    accs: list[float]
        A list with k_times*k_folds size to record the accuracy.
    adv_accs: list[float]
        A list with k_times*k_folds size to record the adversarial accuracy.
    time_costs: list[float]
        A list with k_times*k_folds size to record the cost time.
    samples: array-like of shape (n_samples, n_features)
    adv_samples: list
        The corresponding adversarial samples.
        If the element is None, the corresponding sample has no adversarial samples.
    loss_op: list[int]
        The optimized loss after each splitting during the trainig process. 
        We just record the optimized loss on the last, i.e., k_times*k_folds-th trainig process.
    loss_01: list[int]
        The 0/1 loss after each splitting during the trainig process.
        We just record the 0/1 loss on the last, i.e., k_times*k_folds-th trainig process. 
    """
    max_tree_num = int(opts['--max_tree_num'])
    max_depth = int(opts['--max_depth'])
    min_samples_split = int(opts['--min_samples_split'])
    min_samples_leaf = int(opts['--min_samples_leaf'])
    max_features = opts['--max_features']
    k_times = int(opts['--k_times'])
    k_folds = int(opts['--k_folds'])
    
    
    method = opts['-m']
    if not (opts['--random_seed'] is None):
        random_seed = int(opts['--random_seed'])
    else:
        random_seed = None
    max_attempt_num = int(opts['--max_attempt_num'])

    if_record_loss = '--record_loss' in opts
    if_record_img = '--record_img' in opts
    if_sample = '--if_sample' in opts


    if '--all' in opts:
        path = './out/result/all'
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(path+'/imgs_fail')
            os.mkdir(path+'/imgs_success')
        datasets = load_all()
        epsilon_dict = load_epsilons_dict()
        epsilons = [epsilon_dict[name] for (name,_,_) in datasets]
    else:
        # path = './out/result/'+time.asctime(time.localtime(time.time())).replace(':','_')[4:]
        data_name = opts['--dataset']
        path = './out/result/%s_%s'%(data_name, epsilon)
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(path+'/imgs_fail')
            os.mkdir(path+'/imgs_success')
        # Load the dataset
        datasets = load_dataset(data_name)
        epsilons = [float(opts['--epsilon'])]

    assert method == 'FPRDT' or method == 'PRAdaBoost'


    accs_all = []
    adv_accs_all = []
    time_costs_all = []
    depths_all = []
    leaf_nums_all = []
    samples_all = []
    adv_samples_all = []
    tree_nums_all = []
    data_name_all = []
    loss_op_all = []
    loss_01_all = []

    for dataset,epsilon in zip(datasets, epsilons):
        
        name, X, y = dataset[0],dataset[1], dataset[2]
        X = MinMaxScaler().fit_transform(X)
        data_name_all.append(name)

        accs = []
        adv_accs = []
        time_costs = []
        depths = []
        leaf_nums = []
        samples = []
        adv_samples = []
        tree_nums = []

        for random_state in range(k_times):
            k_folds_cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
            for count, (train_index, test_index) in enumerate(k_folds_cv.split(X, y)):
                # Initialize the model
                if method == 'FPRDT':
                    model = FPRDecisionTree(max_depth, min_samples_split, \
                            min_samples_leaf, max_features, epsilon, random_seed,\
                            if_record_loss)
                else:
                    model = PRAdaBoost(max_tree_num, max_depth, \
                            min_samples_split, min_samples_leaf, epsilon, if_sample, \
                            max_attempt_num, random_seed)
                print('Times:%d Folds:%d'%(random_state, count))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if method == 'FPRDT':
                    depth, leaf_num, tree_num, acc, adv_acc, time_cost, point, adv_point \
                    = assess_tree_method(model, X_train, y_train, X_test, y_test, if_record_loss, if_record_img)
                    # print((model.wrong_leaf_num>0).sum()/model.wrong_leaf_num.shape[0])
                    # choose_feature = model.get_feature()
                    # img = np.zeros([28,28])
                    # for f in choose_feature:
                    #     img[f//28][f%28] = 1
                    # plt.imsave('choosen_feature_%d_%d.png'%(random_state, count), img)
                else:
                    depth, leaf_num, tree_num, acc, adv_acc, time_cost, point, adv_point \
                    = assess_boosting_method(model, X_train, y_train, X_test, y_test, if_record_loss, if_record_img)
                tree_nums.append(tree_num)
                accs.append(acc)
                adv_accs.append(adv_acc)
                time_costs.append(time_cost)
                depths.append(depth)
                leaf_nums.append(leaf_num)
                if if_record_img:
                    samples.append(point)
                    adv_samples += adv_point
        
        if if_record_loss:
            loss_op = [i/X_train.shape[0] for i in model.loss_op]
            loss_01 = [i/X_train.shape[0] for i in model.loss_01]
        else:
            loss_op = None
            loss_01 = None

        accs_all.append(accs)
        adv_accs_all.append(adv_accs)
        time_costs_all.append(time_costs)
        depths_all.append(depths)
        leaf_nums_all.append(leaf_nums)
        tree_nums_all.append(tree_nums)
        loss_op_all.append(loss_op)
        loss_01_all.append(loss_01)

        if if_record_img:
            samples_all.append(np.array(samples).reshape([-1, X.shape[1]]))
            adv_samples_all.append(adv_samples)
        else:
            samples_all.append(None)
            adv_samples_all.append(None)
        
    return data_name_all, depths_all, leaf_nums_all, tree_nums_all, accs_all, adv_accs_all, time_costs_all, samples_all, adv_samples_all, loss_op_all, loss_01_all

def main(opts):
    method = opts['-m']
    if_record_loss = '--record_loss' in opts
    if_record_img = '--record_img' in opts

    if '--all' in opts:
        path = './out/result/all' 
    else:
        epsilon = float(opts['--epsilon'])
        data_name = opts['--method']
        path = './out/result/%s_%s'%(data_name, epsilon)

    # Train and test the model on the loaded dataset
    names, depths, leaf_nums, tree_nums, accs, adv_accs, time_costs, samples, adv_samples, losses_op, losses_01 \
        = train_kfold_model(opts)
    
    write_results_to_files(names, method, depths, leaf_nums, tree_nums, accs, adv_accs, \
        time_costs, path, samples, adv_samples, if_record_img, losses_op, losses_01, \
        if_record_loss)  

    


if __name__ == '__main__':
    opts = getpars()

    if ('--all' in opts):
        main(opts)
    else:
        if not ('--dataset' in opts):
            print('Please input a name of a dataset as \'--dataset [name]\'')
        if not ('--epsilon' in opts):
            print('Please input a value of a perturbation as \'--epsilon [name]\'')
        else:
            main(opts)
    



    
    



