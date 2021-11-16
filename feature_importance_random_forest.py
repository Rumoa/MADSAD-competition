from sklearn.inspection import permutation_importance
import time 

from Script import *

df_train = pd.read_csv("Data/train.csv")
df_test  = pd.read_csv("Data/test.csv")

df_train = df_train.drop("i", axis=1)
y = df_train.pop("y")
df_train.insert(54,"y", y)
df_test = df_test.drop(["y", "i"], axis = 1)


#Call relevant methods until we have results
# analyze_dataset(df_train, df_test)
df_train, df_test = preprocess_data(df_train, df_test)



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

plt.savefig('Images/'+'feature-importance-mdi.pdf')  
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
plt.savefig('Images/'+'feature-importance-permutations-full-model.pdf')  
plt.show()

