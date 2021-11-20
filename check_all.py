from numpy import zeros
from Script import *
import itertools
algorithms = ['Rf', 'svm', 'xgbhist', 'knn']

columns = ['all', 'no cat']

outliers = ['yes', 'no']

onehot = ['yes', 'no']




list_of_options = list(itertools.product(algorithms, columns, outliers, onehot))

acc = np.zeros(len(list_of_options))
keys = ['algorithms', 'columns', 'outliers', 'onehot']

dicc = dict(zip(list_of_options, acc))

print("hola")

