import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats.stats import mode
from scipy import stats
from PIL import Image
from adversary import DecisionTreeAdversary
from toolbox import Model

def convert_numpy(obj):
    """
    Convert numpy ints and floats to python types. Useful when converting objects to JSON.

    Parameters
    ----------
    obj : {np.int32, np.int64, np.float32, np.float64, np.longlong}
        Number to convert to python int or float.
    """
    if (
        isinstance(obj, np.int32)
        or isinstance(obj, np.int64)
        or isinstance(obj, np.longlong)
    ):
        return int(obj)
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    raise TypeError(f"Cannot convert type {type(obj)} to int or float")

def assess_tree_method(model, X_train, y_train, X_test, y_test, record_loss, record_img):
    """
    Assess the model with (X_train, y_train) as training set, (X_test, y_test) as testing set.

    Parameters
    ----------
    model: object including a fit method. 
    X_train: array-like of shape (n_samples, n_features)
    y_train: array-like of shape (n_samples, )
    X_test: array-like of shape (n_samples, n_features)
    y_test: array-like of shape (n_samples, )
    record_loss: bool
        Whether to record the 0/1 loss and the optimized loss in the training process.
    record_img: bool
        Whether to record the imgs and the corresponding adversarial imgs after training process.

    Returns
    ---------
    depth: int
        The depth of the tree after the training.
    leaf_num: int
        The number of the leaves after the training.
    acc: float
        The accuracy on the testing dataset.
    advacc: float
        The adversarial accuracy on the testing dataset.
    time_cost: float
        The time spend on the training process.
    samples: array-like of shape (n_samples, n_features)
    adv_samples: array-like of shape (n_samples, n_features)
        The corresponding adversarial samples.
    """
    begin = time.time()
    model.fit_BFS(X_train, y_train)
    end = time.time()

    # print(model.root_.left_child.left_child)
    # print(model.root_.right_child)

    pre_test_label = model.predict(X_test)
    acc = (y_test==pre_test_label).sum()/y_test.shape[0]

    time_cost = end - begin

    attack_model = [model.epsilon]*X_test.shape[0]
    attacker = DecisionTreeAdversary(model, 'ours', attack_model, [True] * X_test.shape[1], None, False)
    adv_acc = attacker.adversarial_accuracy(X_test, y_test)

    if record_img:
        acc_index = (y_test==pre_test_label)
        X_adv_test = []
        for x,y in zip(X_test[acc_index], y_test[acc_index]):
            X_adv_test.append(attacker.get_adversarial_examples(x, y))
        return model.get_depth(), model.get_leaf_number(), 1, acc, adv_acc, time_cost, X_test[acc_index], X_adv_test
    else:
        return model.get_depth(), model.get_leaf_number(), 1, acc, adv_acc, time_cost, None, None


def assess_boosting_method(model, X_train, y_train, X_test, y_test, record_loss, record_img):
    begin = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    pre_test_label = model.predict(X_test)
    acc = (y_test==pre_test_label).sum()/y_test.shape[0]

    time_cost = end - begin

    depths, leaf_numbers = model.get_depth(), model.get_leaf_number()

    n_trees = len(model.estimators_)

    epsilon = model.epsilon

    model.to_xgboost_json('./out/model/PRAdaBoost.json')
    model = Model.from_json_file('./out/model/PRAdaBoost.json', 2)

    adv_acc = model.adversarial_accuracy(X_test, y_test, epsilon=epsilon)
    
    if record_img:
        acc_index = (y_test==pre_test_label)
        options = {"n_threads": 6}
        X_adv_test = []
        for x,y in zip(X_test[acc_index][:300], y_test[acc_index][:300]):
            X_adv_test.append(model.adversarial_examples(x.reshape(1,-1), [y], options=options))
        return depths, leaf_numbers, n_trees, acc, adv_acc, time_cost, X_test[acc_index][:300], X_adv_test
    else:
        return depths, leaf_numbers, n_trees, acc, adv_acc, time_cost, None, None


def write_txtfile(dataset, method, performance, outpath):
    """
    Write the performance of method on dataset to outpath.

    Parameters
    ----------
    dataset: string
        The name of the dataset to be written.
    method: string
        The name of the method.
    performance: list[float]
        The performance to be written.
    outpath: string
        The path of the written files.

    """
    with open(outpath, 'a') as f:
        f.write(f'{dataset} ')
        f.write(f'{method} ')
        f.write('%.4f+-%.4f '%(np.mean(performance), np.std(performance,ddof=1)))
        f.write(f'{performance}\n')


def write_results_to_files(names, method, depth, leaf_num, tree_num, acc, adv_acc, time_cost, path, \
    samples=None, adv_samples=None, record_img=False, loss_op=[], loss_01=[], record_loss=False):
    """
    Write all revelent results to files.

    """

    for i in range(len(names)):
        write_txtfile(names[i], method, acc[i], path+'/result_acc.txt')
        write_txtfile(names[i], method, adv_acc[i], path+'/result_advacc.txt')
        write_txtfile(names[i], method, time_cost[i], path+'/result_timecost.txt')
        write_txtfile(names[i], method, depth[i], path+'/result_depth.txt')
        write_txtfile(names[i], method, leaf_num[i], path+'/result_leaf_num.txt')
        write_txtfile(names[i], method, tree_num[i], path+'/result_tree_num.txt')


    if record_loss:
        for i in range(len(names)):
            plt.clf()
            plt.plot([i for i in range(len(loss_01[i]))], loss_01[i])
            plt.savefig(fname = path+'/'+names[i]+'_'+method+'.png')

    if record_img:
        for i in range(len(names)):
            length = int(samples[i][0].shape[0]**0.5+0.1)
            for count, (point, adv_point) in enumerate(zip(samples[i], adv_samples[i])):
                if adv_point is None:
                    point = point*255
                    point = point.astype(np.uint8)
                    im = Image.fromarray(point.reshape([length,-1]))
                    im = im.convert('RGB')
                    im.save(path+'/imgs/'+method+'/fail/'+str(count)+'.png')
                else:
                    point = point*255
                    point = point.astype(np.uint8)
                    adv_point = adv_point*255
                    adv_point = adv_point.astype(np.uint8)
                    im = Image.fromarray(point.reshape([length,-1]))
                    im = im.convert('RGB')
                    im.save(path+'/imgs/'+method+'/success/'+str(count)+'.png')
                    im = Image.fromarray(adv_point.reshape([length,-1]))
                    im = im.convert('RGB')
                    im.save(path+'/imgs/'+method+'/success/'+str(count)+'_adv.png')


