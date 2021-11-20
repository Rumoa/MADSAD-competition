import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.svm import SVC

from scipy import stats


def createpipe(options, oh_ec):
    pipe = []
    algorithm = options[0]
    onehot = options[3]
    need_normalization = False
    if algorithm=="svm":
        need_normalization=True

    if onehot == "yes":
        pipe.append(("oh encoder", oh_ec))

    if need_normalization == True:
        pipe.append(("scaler",StandardScaler()))

    pipe.append((algorithm ,choose_alg(algorithm)))
    pipeline = Pipeline(pipe)
    return pipeline

def choose_alg(select_algorithm):
    if select_algorithm == "Rf":
        alg = RandomForestClassifier()
    if select_algorithm == "svm":
        alg =  SVC()
    if select_algorithm ==  "xgbhist":
        alg = HistGradientBoostingClassifier()     
    return alg       

def drop_col(train, test, cols):
    trainc = train.copy()
    testc = test.copy()
    trainp = trainc.drop(cols, axis=1)
    testp = testc.drop(cols, axis=1)
    return trainp, testp

def drop_cat(train, test):
    col_c = list(train.filter(like='c', axis=1).columns)
    if len(col_c) != 0:
        trainp, testp = drop_col(train, test, col_c)
    else:
        trainp, testp = train, test

    return trainp, testp

def drop_low_unique(train, test):
    columns_to_drop = train.columns[train.apply(lambda col: train.nunique()[col.name] < 100 and col.name != 'y' and 'c' not in col.name and 'o' not in col.name)]
    trainp, testp = drop_col(train, test, columns_to_drop)
    return trainp, testp

def drop_low_importance(train, test):
    drop_column = ["x38", "x39", "c4", "c5", "c7", "c1", "c2", "c3", "c6", "c8", "c9","c10","c11","c12"]
    trainp, testp = drop_col(train, test, drop_column)
    return trainp, testp

def drop_outliers(train, test, remove_also_test = False):
    col_x = list(train.filter(like='x', axis=1).columns)
    df_train_NO_OUTLIERS = train.copy()
    aux = train[(np.abs(stats.zscore(train[col_x])) < 3)]
    df_train_NO_OUTLIERS[col_x] = aux[col_x]


    for i in col_x:
        df_train_NO_OUTLIERS[i].fillna(value=df_train_NO_OUTLIERS[i].mean(), inplace=True)

    ptest = test.copy()
    if remove_also_test:
        aux = test[(np.abs(stats.zscore(test[col_x])) < 3)]
        ptest[col_x] = aux[col_x]

        for i in col_x:
            ptest[i].fillna(value=ptest[i].mean(), inplace=True)
    return df_train_NO_OUTLIERS, ptest  


def select_data(train, test, options, remove_test_outliers = False):
    ntrain = train.copy()
    ntest = test.copy()
    columns = options[1]
    outliers = options[2]
    include_low_importance_cols = options[4]
    include_low_unique = options[5]

    if outliers == "no":
        ntrain, ntest = drop_outliers(ntrain, ntest, remove_test_outliers)
    

    if include_low_importance_cols == 'no':
        ntrain, ntest = drop_low_importance(ntrain, ntest)

    if include_low_unique == "no":
        ntrain, ntest = drop_low_unique(ntrain, ntest)

    if columns == 'no cat':
        ntrain, ntest = drop_cat(ntrain, ntest)
    
    
    return ntrain, ntest          


def train_with_options(train, test, options, ncv=3):
    ntrain, ntest = select_data(train, test, options)
    X = ntrain.iloc[:, 0:-1]
    y = ntrain.iloc[:, -1]
    
    one_hot_encoder = make_column_transformer((
                                                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                                make_column_selector(dtype_include="category"),
                                              ),
                                              remainder="passthrough")
    model_pipe = createpipe(options, one_hot_encoder)
    hard_scores = cross_val_score(model_pipe, X, y, cv=ncv)
    return hard_scores, hard_scores.mean()
