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

X = df_train.iloc[:, 0:-1]
y = df_train.iloc[:, -1]



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)


feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=1)
forest.fit(X_train, y_train)

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.show()



start_time = time.time()
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)



fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()