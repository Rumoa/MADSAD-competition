#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to generate the results for the KDD Kaggle competition.

@authors: António, Gabriel
"""

#General Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocessing libraries
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#ML libraries
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def analyze_dataset(df_train, df_test):
    """
    Method used to analyze the dataset and plot useful data BEFORE messing with the data and
    using any type of machine learning

    Parameters
    ----------
    df_train : Pandas Dataframe
        Contains all the examples that will be used when training the algorithm
    df_test : Pandas Dataframe
        Contains all the examples that will be used when testing the algorithm, does not include "y" column

    Returns
    -------
    None.

    """
    #Print number of missing values for each column
    for i in range(df_train.shape[1]):
        print("Column: ", df_train.columns[i], "NAs: ", df_train.iloc[:, i].isnull().sum())
    
    #Check data imbalancement
    print("-----------------------------------")
    print(df_train["y"].value_counts())
    print("Percentage of minority class: " , df_train["y"].value_counts()[1]/sum(df_train["y"].value_counts()))
    
    #Create a correlation matrix heatmap
    #We should add a way to check correlation between the Y and each variable, right now we are only looking at the X variables
    corr = df_train.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap="Greens",annot=False)
    
    
def preprocess_data(df_train, df_test):
    """
    Method that will be used for all the data preprocessing steps.

    Parameters
    ----------
    df_train : Pandas Dataframe
        Contains all the examples that will be used when training the algorithm
    df_test : Pandas Dataframe
        Contains all the examples that will be used when testing the algorithm, does not include "y" column

    Returns
    -------
    df_train_noNAs : Pandas Dataframe
        A transformed version of the initial pandas dataframe after all the preprocessing steps have been done
    df_test_noNAs : Pandas Dataframe
        A transformed version of the initial pandas dataframe after all the preprocessing steps have been done

    """
    #df_train = df_train.drop("y", axis=1) # Should we keep this? I don't think so
    
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(df_train)
    
    df_train_noNAs=pd.DataFrame(imp_mean.fit_transform(df_train))
    df_train_noNAs.columns=df_train.columns
    df_train_noNAs.index=df_train.index
    
    df_test_noNAs=pd.DataFrame(imp_mean.fit_transform(df_test))
    df_test_noNAs.columns=df_test.columns
    df_test_noNAs.index=df_test.index
    
    for i in range(df_train_noNAs.shape[1]):
        print("Column ", df_train_noNAs.columns[i], "NAs: ", df_train_noNAs.iloc[:, i].isnull().sum())
    
    return df_train_noNAs, df_test_noNAs

def compute_metrics(y_pred, y_real):
    """
    

    Parameters
    ----------
    y_pred : List
        Results predicted by the algorithm
    y_real : List
        True values

    Returns
    -------
    metrics : String
        String ready to print with f1, precision, recall and accuracy values

    """
    metrics = {"F1: ": f1_score(y_pred, y_real),
                "Precision: " : precision_score(y_pred, y_real),
                "Recall: " : recall_score(y_pred, y_real),
                "Accuracy: ":  accuracy_score(y_pred, y_real)}
    return metrics

def cv_metrics_summary(model, X, y, no_cv=10, scoring=['f1','precision', 'recall', 'accuracy' ]):
    metrics_cv = pd.DataFrame(cross_validate(model, X, y, cv=no_cv, scoring=scoring))
    metrics_cv.loc['mean'] = metrics_cv.mean()
    return metrics_cv

def save_predictions(preds, name = "preds"):
    """
    Saves the predictions in a format that allows them to be uploaded to the kaggle competition.

    Parameters
    ----------
    preds : List
        Contains the predictions obtained by the machine learning algorithms
    name : String, optional
        Name with which to save the predictions file. The default is "preds".

    Returns
    -------
    None.

    """
    index = pd.read_csv("Data/test.csv").loc[ :, "i"]
    prediction = pd.concat([index, pd.DataFrame(preds).astype(int)] , axis=1)
    prediction.columns = ["i", "y"]
    prediction.to_csv(name + ".csv", index=False)
    #return prediction
if __name__ == '__main__':

    #Read and reshape the data initially
    df_train = pd.read_csv("Data/train.csv")
    df_test  = pd.read_csv("Data/test.csv")

    df_train = df_train.drop("i", axis=1)
    y = df_train.pop("y")
    df_train.insert(54,"y", y)
    df_test = df_test.drop(["y", "i"], axis = 1)

    #Changing variable data types
    df_test.loc[:, "c1":"c12"] = df_test.loc[:, "c1":"c12"].astype("category")
    df_test.loc[:, ["o1", "o2"]] = df_test.loc[:, ["o1", "o2"]].astype("category")
    df_train.loc[:, "c1":"c12"] = df_train.loc[:, "c1":"c12"].astype("category")
    df_train.loc[:, ["o1", "o2", "y"]] = df_train.loc[:, ["o1", "o2", "y"]].astype("category")

    #Call relevant methods until we have results
    analyze_dataset(df_train, df_test)
    df_train, df_test = preprocess_data(df_train, df_test)


    # HERE JUST DO WHATEVER YOU WANT WITH THE MODELS. I DID A STRATIFIED K-FOLD ON A RANDOM FOREST
    # COMMENT OR DELETE WHATEVER YOU DON'T WANT TO USE AND MAKE YOUR OWN MODELS TO TEST AND ALL THAT SHIT

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # skf = StratifiedKFold(n_splits = 3)
    # score_list = []
    # models = []
    # #Stratified fold to keep the data imbalance while training
    # #Train 3 times to simualte a 3 way cross-validation
    # for train, test in skf.split(df_train, y):
    #     print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))
    #     train_x = df_train.iloc[train]
    #     train_y = y.iloc[train]
    #     test_x = df_train.iloc[test]
    #     test_y = y.iloc[test]
    #     #CREATE THE MODELS HERE IF POSSIBLE
        
    #     #Random Forest
    #     clf = RandomForestClassifier(random_state = 0)
    #     clf.fit(train_x, train_y)
    #     y_pred = clf.predict(test_x)
    #     print(compute_metrics(y_pred, test_y))
        
    #     #SAVE BEST SCORES AND BEST MODELS TO TEST ON THE DATA LATER.
    #     score_list.append(clf.score(test_x,test_y))
    #     models.append(clf)

    # #Check Metrics attained
    # best_model_index = score_list.index(max(score_list))
    # preds = models[best_model_index].predict(df_test)

    estimators = []
    model1 = RandomForestClassifier()
    estimators.append(('Random Forest', model1))
    model2 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    estimators.append(('svm', model2))
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    results = cv_metrics_summary(ensemble, df_train, y)
    # results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
    print(results)
    ensemble.fit(df_train, y)
    preds = ensemble.predict(df_test)

    #Save results
    save_predictions(preds)

