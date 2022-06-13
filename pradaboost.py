from fprdt import FPRDecisionTree
import numpy as np
import json

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    else:
        return np.random.RandomState(seed)

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


class PRAdaBoost:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        epsilon=0,
        if_sample=False,
        max_attempt_num=30,
        random_seed=None,
    ):
        """
        Parameters
        ----------
        n_estimators : int
            The maximum number of tree.
        max_depth : int, optional
            The maximum depth for the decision tree once fitted.
        min_samples_split : int, optional
            The minimum number of samples required to split a node.
        min_samples_leaf : int, optional
            The minimum number of samples required to make a leaf.
        epsilon : float, optional
            The value of the perturbation in L_\infinity attack.
        if_sample : bool, optional
            The value to indicate we use the policy of sampling or reweighting when
            fitting a new distribution.
        random_seed : int, optional
            The value of the random seed used for sampling from training set.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.epsilon = epsilon
        self.if_sample = if_sample
        self.random_seed = random_seed
        self.max_attempt_num = max_attempt_num
        if self.if_sample:
            self.random_state = check_random_state(self.random_seed)
              

    def fit(self, X, y):
        self.n_samples_, self.n_features_ = X.shape
        self.estimators_ = []
        self.alpha = []
        weights = np.ones(X.shape[0])*(1/X.shape[0])
        print('')
        for i in range(self.n_estimators):
            print('\u001b[1A\u001b[100DTree:%d building...'%i)
            if self.if_sample:
                for _  in range(self.max_attempt_num):
                    tree = FPRDecisionTree(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        epsilon=self.epsilon
                    )
                    # Sample from the distribution.
                    choosen_index = self.random_state.choice(
                        [i for i in range(self.n_samples_)], self.n_samples_, \
                            replace=True, p=weights
                    )
                    tree.fit(X[choosen_index], y[choosen_index])
                    tree.y = None
                    tree.wrong_leaf_num = None
                    # Compute the adversarial error
                    attacked = tree.adversarial_attack(X, y)
                    e_t = 0
                    for (w, a) in zip(weights, attacked):
                        if a:
                            e_t += w
                    # print(e_t)
                    if e_t < 0.49:
                        self.estimators_.append(tree)
                        alpha_t = 0.5*np.log((1-e_t)/(e_t))
                        self.alpha.append(alpha_t)
                        for (i,a) in enumerate(attacked):
                            if a:
                                weights[i] *= np.e**(alpha_t)
                            else:
                                weights[i] *= np.e**(-alpha_t)
                        weights /= weights.sum()
                        break
                else:
                    break
            else:
                tree = FPRDecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    epsilon=self.epsilon
                )
                tree.fit(X, y, weights)
                
                tree.y = None
                e_t = 0
                for i in range(X.shape[0]):
                    if tree.wrong_leaf_num[i] >= 1:
                        e_t += weights[i]
                if e_t >= 0.5:
                    break
                self.estimators_.append(tree)  
                alpha_t = 0.5*np.log((1-e_t)/(e_t))

                for i in range(X.shape[0]):
                    if tree.wrong_leaf_num[i] >= 1:
                        weights[i] *= np.e**(alpha_t)
                    else:
                        weights[i] *= np.e**(-alpha_t)
                self.alpha.append(alpha_t)
                tree.wrong_leaf_num = None
                weights/=weights.sum()

    def get_depth(self):
        depth = []
        for estimator in self.estimators_:
            depth.append(estimator.get_depth())
        return sum(depth)/len(depth)

    def get_leaf_number(self):
        leaf_number = []
        for estimator in self.estimators_:
            leaf_number.append(estimator.get_leaf_number())
        return sum(leaf_number)/len(leaf_number)

    def predict(self, X):
        """
        Predict the classes of the input samples X.
        The predicted class is the rounded average of the class labels in
        each predicted leaf.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels
        """
        # predictions = []
        # for x in X:
        #     sub_pre = 0
        #     for count, tree in enumerate(self.estimators_):
        #         sub_predict = tree.predict(x)
        #         sub_predict = sub_predict[1]*2-1
        #         sub_pre += sub_predict
        #     predictions.append((sub_pre>0)*1)
        # return np.array(predictions)
        predictions = np.zeros(X.shape[0])
        for count, tree in enumerate(self.estimators_):
            sub_predict = tree.predict(X)
            sub_predict = sub_predict*2-1
            predictions += self.alpha[count]*sub_predict
        return (predictions>0)*1

    def __str__(self):
        result = ""
        result += f"Parameters: {self.get_params()}\n"

        if hasattr(self, "estimators_"):
            for tree in self.estimators_:
                result += f"Tree:\n{tree.root_.pretty_print()}\n"
        else:
            result += "Forest has not yet been fitted"
        return result

    def to_json(self, output_file="forest.json"):
        with open(output_file, "w") as fp:
            dictionary = {
                "params": self.get_params(),
            }
            if hasattr(self, "estimators_"):
                dictionary["trees"] = [tree.to_json(None) for tree in self.estimators_]
                json.dump(dictionary, fp, indent=2, default=convert_numpy)
            else:
                dictionary["trees"] = None
                json.dump(dictionary, fp)

    def to_xgboost_json(self, output_file="forest.json"):
        if hasattr(self, "estimators_"):
            dictionary = [tree.to_xgboost_json(self.alpha[i], None) for i,tree in enumerate(self.estimators_)]

            if output_file:
                with open(output_file, "w") as fp:
                    json.dump(dictionary, fp, indent=2, default=convert_numpy)
            else:
                return dictionary
        else:
            raise Exception("Forest not yet fitted")