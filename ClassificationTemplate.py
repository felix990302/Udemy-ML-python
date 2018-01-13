# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("Social_Network_Ads.csv").iloc[:, 2:].values

# missing data
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
dataset[:, 1:3] = imputer.fit_transform(dataset[:, 1:3])
'''

# encode categorical data
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
dataset[:, 0] = labelencoder.fit_transform(dataset[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
dataset[:, 3] = labelencoder.fit_transform(dataset[:, 3])

dataset = onehotencoder.fit_transform(dataset).toarray()
'''

# spliting into training, CV, test sets
np.random.shuffle(dataset)
train, test = np.split(dataset, [int(.8 * len(dataset))])
X_train = train[:, :-1]
y_train = train[:, -1]
X_test = test[:, :-1]
y_test = test[:, -1]

# feature scaling / mean normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fit Logistic Regression
from sklearn.linear_model import LogisticRegression
clas = LogisticRegression(penalty="l2",  
                          dual=True, 
                          C=1,fit_intercept=True,
                          solver='liblinear',
                          multi_class='ovr',
                          )
clas.fit(X_train,y_train)

# recursive feature elimination
# from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=clas, 
              step=1, 
              cv=10,
              scoring='f1_micro',
              n_jobs=-1)
rfecv.fit(X_train, y_train)

# relev_feat = rfecv.support_
relev_feat = rfecv.ranking_ == 1

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# limit features
X_train = X_train[:,relev_feat]
X_cv = X_cv[:,relev_feat]
X_test = X_test[:,relev_feat]


# add features
'''
from sklearn.preprocessing import PolynomialFeatures

poly_feat = PolynomialFeatures(degree = 1)
X_train = poly_feat.fit_transform(X_train)
X_cv = poly_feat.fit_transform(X_cv)
X_test = poly_feat.fit_transform(X_test)
'''

# Predict test set results
'''
y_pred = clas.predict(X_test)
'''

# Confusion Matrix
'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
precision = cm[1,1]/(cm[1,1]+cm[0,1])
recall = cm[1,1]/(cm[1,1]+cm[1,0])
f1 = 2*(precision*recall)/(precision+recall)
print("f1:", f1)
'''

# Applying k-Fold Cross Validation
'''
from sklearn.model_selection import cross_val_score
f1 = cross_val_score(estimator=clas, X=X_train, y=y_train, scoring='f1', cv=10, )
print(f1.mean())
print(f1.std())
'''

# Applyig Grid Search to find best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[0.125,0.25,0.375,0.5,0.675,0.75,0.875,1], 'dual':[True], 'random_state':[42]}
             ]
grid_search = GridSearchCV(estimator=clas,
                           param_grid = parameters,
                           scoring = 'f1',
                           cv=20,
                           n_jobs=-1)
grid_search = grid_search.fit(X_train,y_train)

print("best f1 score:", grid_search.best_score_)
best_params = grid_search.best_params_


clas = grid_search.best_estimator_

# Learning Curve
from sklearn.model_selection import learning_curve
plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
train_sizes, train_scores, test_scores = learning_curve(clas,
                                                        X_train, 
                                                        y_train, 
                                                        scoring='f1_micro',
                                                        cv=10, 
                                                        n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.legend(loc="best")


# Visualize test set
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X_1, X_2= np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step=0.01),
                      np.arange(start=X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step=0.01))
plt.contourf(X_1,X_2, clas.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),
    alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X_1.min(), X_1.max())
plt.ylim(X_2.min(), X_2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0], X_set[y_set==j,1],
                c = ListedColormap(('red','green'))(i), label = j)

plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# predict test set
y_pred = clas.predict(X_test)

# scoring

from sklearn import metrics
final_score = metrics.f1_score(y_test, y_pred, average='micro')


