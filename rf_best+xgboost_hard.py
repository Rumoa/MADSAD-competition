from sklearn.inspection import permutation_importance
import time 

from Script import *

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
# #Train 3 times to simulate a 3 way cross-validation
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

X = df_train.iloc[:, 0:-1]
y = df_train.iloc[:, -1]


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier



import joblib 
rf_grid_search = joblib.load("Temp/rf_grid_search.joblib")

clf1 = rf_grid_search.best_estimator_



clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0)

print("Fitting Classifiers")
eclf1 = VotingClassifier(estimators=[
        ('rf', clf1), ('xgb', clf2)], voting='hard')






print("Hard Voting ")
hard_scores = cross_val_score(eclf1, X, y, cv=10)

print("aux stop")


import joblib
name_of_model = "rf_best+xgboost"

joblib.dump(hard_scores, "Temp/"+name_of_model+"_hard_scores"+".joblib")


print("hard voting", hard_scores.mean())


print("aux stop")

# joblib.dump(eclf1, "Temp/"+name_of_model+".joblib")

eclf1.fit(X, y)

preds_hard = eclf1.predict(df_test)
save_predictions(preds_hard, "Predictions/"+name_of_model+"_hard")





