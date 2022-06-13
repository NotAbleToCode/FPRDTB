import numpy as np
from sklearn.datasets import fetch_openml
from collections import defaultdict

def load_epsilons_dict(epsilon=0.1):
    epsilons = defaultdict(lambda: epsilon)
    epsilons["cod-rna"] = 0.025
    epsilons["diabetes"] = 0.05
    epsilons["wine"] = 0.05
    epsilons["spambase"] = 0.05
    epsilons["ionosphere"] = 0.2
    epsilons["breast-cancer"] = 0.3
    epsilons["MNIST 2 vs 6"] = 0.4
    epsilons["MNIST 1 vs 7"] = 0.4
    epsilons["MNIST 3 vs 8"] = 0.4
    epsilons["Fashion-MNIST 7 vs 9"] = 0.2
    epsilons["Fashion-MNIST 3 vs 4"] = 0.2
    epsilons["Fashion-MNIST 2 vs 5"] = 0.2

    epsilons["steel-plates-fault"] = 0.1
    epsilons["phoneme"] = 0.01
    epsilons["ozone-level-8hr"] = 0.2
    epsilons["madelon"] = 0.005
    epsilons["wilt"] = 0.1
    epsilons["qsar-biodeg"] = 0.05
    epsilons["hill-valley"] = 0.00
    epsilons["scene"] = 0.3
    epsilons["gina_agnostic"] = 0.05

    epsilons["wall-robot-navigation"] = 0.1
    epsilons["GesturePhaseSegmentationProcessed"] = 0.01
    epsilons["artificial-characters"] = 0.1
    epsilons["har"] = 0.1
    epsilons["JapaneseVowels"] = 0.1

    epsilons["pollen"] = 0.00
    epsilons["texture"] = 0.2
    epsilons["delta_ailerons"] = 0.2
    return epsilons

def load_wine():
    # Refered to as 'wine'
    data = fetch_openml("wine_quality", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    is_numeric = [True]*11
    y = np.where(y >= 6, 0, 1)  # Larger or equal to a 6 is a 'good' wine
    return (
        "wine",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_diabetes():
    # Refered to as 'diabetes'
    data = fetch_openml("diabetes", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == "tested_negative", 0, 1)
    is_numeric = [True]*8
    return (
        "diabetes",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_ionosphere():
    # Refered to as 'ionosphere'
    data = fetch_openml("ionosphere", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == "b", 1, 0)
    is_numeric = [True]*34
    return (
        "ionosphere",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_ijcnn():
    # Refered to as 'ijcnn'
    data = fetch_openml("ijcnn", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    X = X.toarray()
    y = np.where(y == -1, 0, 1)
    is_numeric = [True]*22
    return (
        "ijcnn",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_spambase():
    # Refered to as 'spambase'
    data = fetch_openml("spambase", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = y.astype(int)
    is_numeric = [True]*57
    return (
        "spambase",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_diabetes():
    # Refered to as 'diabetes'
    data = fetch_openml("diabetes", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == "tested_positive", 1, 0)
    y = y.astype(int)
    is_numeric = [True]*8
    return (
        "diabetes",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_balance_scale():
    # Refered to as 'balance-scale'
    data = fetch_openml("balance-scale", version=2, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == "P", 1, 0)
    y = y.astype(int)
    is_numeric = [True]*8
    return (
        "balance-scale",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_banknote_authentication():
    # Refered to as 'banknote-authentication'
    data = fetch_openml(
        "banknote-authentication", version=1, return_X_y=False, as_frame=False
    )
    X = data.data
    y = data.target
    y = np.where(y == "2", 1, 0)
    is_numeric = [True]*4
    return (
        "banknote-authentication",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_cylinder_bands():
    # Refered to as 'cylinder-bands'
    data = fetch_openml("cylinder-bands", version=2, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == "band", 1, 0)
    y = y.astype(int)
    # Remove rows with missing values
    y = y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]
    is_numeric = [True] * 37
    return (
        "cylinder-bands",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_blood_transfusion():
    # Refered to as 'blood-transfusion'
    data = fetch_openml(
        "blood-transfusion-service-center", version=1, return_X_y=False, as_frame=False
    )
    X = data.data
    y = data.target
    y = np.where(y == "2", 1, 0)
    y = y.astype(int)
    is_numeric = [True] * 4
    return (
        "blood-transfusion",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_climate_model_simulation():
    # Refered to as 'climate-model-simulation'
    data = fetch_openml(
        "climate-model-simulation-crashes", version=4, return_X_y=False, as_frame=False
    )
    X = data.data
    y = data.target
    y = y.astype(int)
    is_numeric = [True] * 18
    return (
        "climate-model-simulation",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_sonar():
    # Refered to as 'sonar'
    data = fetch_openml("sonar", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == "Mine", 1, 0)
    y = y.astype(int)
    is_numeric = [True] * 60
    return (
        "sonar",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_haberman():
    # Refered to as 'haberman'
    data = fetch_openml("haberman", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == "2", 1, 0)
    y = y.astype(int)
    is_numeric = [True] * 3
    return (
        "haberman",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_parkinsons():
    # Refered to as 'parkinsons'
    data = fetch_openml("parkinsons", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == "2", 1, 0)
    y = y.astype(int)
    is_numeric = [True] * 22
    return (
        "parkinsons",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_planning_relax():
    # Refered to as 'planning-relax'
    data = fetch_openml("planning-relax", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == "2", 1, 0)
    y = y.astype(int)
    is_numeric = [True] * 12
    return (
        "planning-relax",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_SPECTF():
    # Refered to as 'SPECTF'
    data = fetch_openml("SPECTF", version=2, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = y.astype(int)
    is_numeric = [True] * 44
    return (
        "SPECTF",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_breast_cancer():
    # Refered to as 'breast-cancer'
    data = fetch_openml("breast-w", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == "malignant", 1, 0).astype(int)
    y = y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]
    is_numeric = [True] * 10
    return (
        "breast-cancer",
        X,
        y,
        is_numeric,
        data.categories,
    )


def load_mnist(a, b):
    # Refered to as 'MNIST'
    data = fetch_openml("mnist_784", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = y.astype(int)
    is_numeric = [True] * 784
    X = X[(y == a) | (y == b)]
    y = y[(y == a) | (y == b)]
    y = np.where(y == b, 1, 0)
    # print((y==0).sum())
    # print((y==1).sum())
    return (
        "MNIST %d vs %d"%(a,b),
        X,
        y,
        is_numeric,
        data.categories,
    )
    
def load_fmnist(a,b):
    # Refered to as 'MNIST'
    data = fetch_openml("Fashion-MNIST", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = y.astype(int)
    is_numeric = [True] * 784
    X = X[(y == a) | (y == b)]
    y = y[(y == a) | (y == b)]
    y = np.where(y == b, 1, 0)
    return (
        "Fashion-MNIST %d vs %d"%(a,b),
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_steel_plates_fault():
    data = fetch_openml("steel-plates-fault",  version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 33
    return (
        "steel-plates-fault",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_phoneme():
    data = fetch_openml("phoneme",  version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 5
    return (
        "phoneme",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_ozone_level_8hr():
    data = fetch_openml("ozone-level-8hr", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 72
    return (
        "ozone-level-8hr",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_madelon():
    data = fetch_openml("madelon", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 500
    return (
        "madelon",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_wilt():
    data = fetch_openml("wilt", version=2, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 5
    return (
        "wilt",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_Satellite():
    data = fetch_openml("Satellite", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = y.astype(int)
    is_numeric = [True] * 36
    return (
        "Satellite",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_qsar_biodeg():
    data = fetch_openml("qsar-biodeg", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 41
    return (
        "qsar-biodeg",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_hill_valley():
    data = fetch_openml("hill-valley", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 100
    return (
        "hill-valley",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_scene():
    data = fetch_openml("scene", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 299
    return (
        "scene",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_gina_agnostic():
    data = fetch_openml("gina_agnostic", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 970
    return (
        "gina_agnostic",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_GesturePhaseSegmentationProcessed():
    data = fetch_openml("GesturePhaseSegmentationProcessed", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    X = X[(y == 'D') | (y == 'P')]
    y = y[(y == 'D') | (y == 'P')]
    y = np.where(y == 'D', 1, 0)
    is_numeric = [True] * 32
    return (
        "GesturePhaseSegmentationProcessed",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_wall_robot_navigation():
    data = fetch_openml("wall-robot-navigation", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    X = X[(y == '1') | (y == '2')]
    y = y[(y == '1') | (y == '2')]
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 24
    return (
        "wall-robot-navigation",
        X,
        y,
        is_numeric,
        data.categories,
    )

# C vs G
def load_artificial_characters():
    data = fetch_openml("artificial-characters", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    X = X[(y == '2') | (y == '6')]
    y = y[(y == '2') | (y == '6')]
    y = np.where(y == '2', 1, 0)
    is_numeric = [True] * 7
    return (
        "artificial-characters",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_har():
    data = fetch_openml("har", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    X = X[(y == '1') | (y == '2')]
    y = y[(y == '1') | (y == '2')]
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 561
    return (
        "har",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_JapaneseVowels():
    data = fetch_openml("JapaneseVowels", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    X = X[(y == '3') | (y == '4')]
    y = y[(y == '3') | (y == '4')]
    X = X[:,:8]
    y = np.where(y == '3', 1, 0)
    is_numeric = [True] * 14
    return (
        "JapaneseVowels",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_pollen():
    data = fetch_openml("pollen", version=2, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == 'P', 1, 0)
    is_numeric = [True] * 5
    return (
        "pollen",
        X,
        y,
        is_numeric,
        data.categories,
    )
    
def load_texture():
    data = fetch_openml("texture", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    X = X[(y == '1') | (y == '2')]
    y = y[(y == '1') | (y == '2')]
    y = np.where(y == '1', 1, 0)
    is_numeric = [True] * 40
    return (
        "texture",
        X,
        y,
        is_numeric,
        data.categories,
    )

def load_delta_ailerons():
    data = fetch_openml("delta_ailerons", version=1, return_X_y=False, as_frame=False)
    X = data.data
    y = data.target
    y = np.where(y == 'P', 1, 0)
    is_numeric = [True] * 5
    return (
        "delta_ailerons",
        X,
        y,
        is_numeric,
        data.categories,
    )
    
def load_cifar(a,b):
    from keras.datasets import cifar10
    (X1,y1),(X2,y2) = cifar10.load_data()
    X1 = X1.reshape([50000,-1])
    X2 = X2.reshape([10000,-1])
    X = np.vstack([X1,X2])
    y = np.vstack([y1,y2]).reshape([-1])
    y = y.astype(int)
    is_numeric = [True] * 3072
    X = X[(y == a) | (y == b)]
    y = y[(y == a) | (y == b)]
    y = np.where(y == b, 1, 0)
    return (
        f'cifar10 {a} vs {b}',
        X,
        y,
        is_numeric,
    )

def load_all():
    return [
        load_ionosphere()[:3],
        load_breast_cancer()[:3],
        load_diabetes()[:3],
        load_banknote_authentication()[:3],
        load_JapaneseVowels()[:3],
        load_har()[:3],
        load_spambase()[:3],
        load_GesturePhaseSegmentationProcessed()[:3],
        load_wine()[:3],
        load_mnist(2, 6)[:3],
        load_mnist(3, 8)[:3],
        load_mnist(1, 7)[:3],
        load_fmnist(2, 5)[:3],
        load_fmnist(3, 4)[:3],
        load_fmnist(7, 9)[:3],
        load_cifar(0,5)[:3],
        load_cifar(0,6)[:3],
        load_cifar(4,8)[:3]

        # load_haberman()[:3],
        # load_blood_transfusion()[:3],
        # load_planning_relax()[:3],
        # load_cylinder_bands()[:3],
        # load_SPECTF()[:3],
        # load_parkinsons()[:3],
        # load_sonar()[:3],
        # load_climate_model_simulation()[:3],
        # load_fashion_mnist(1, 2)[:3],
        # load_wall_robot_navigation()[:3],
        # load_artificial_characters()[:3],

        # load_fashion_mnist(i,j+5)[:3] for i in range(5) for j in range(5)
        # load_mnist([0,1,2],[3,4,5])[:3]
        # load_cifar_10(i,j+5)[:3] for i in range(5) for j in range(5)
        # load_cifar_10(4,8)[:3]
    ]

def load_dataset(dataset):
    if 'mnist' in dataset or 'fmnist' in dataset or 'cifar' in dataset:
        class1 = dataset[-3]
        class2 = dataset[-1]
        return [eval('load_%s(%s,%s)[:3]'%(dataset[:-3], class1, class2))]
    else:
        return [eval('load_%s()[:3]'%dataset)]

if __name__ == '__main__':
    dataset = load_dataset('mnist2v3')
    print(dataset[1].shape)