from sklearn import svm
from sklearn import datasets
import joblib

# create model
clf = svm.SVC(probability=True)
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)

# save model
joblib.dump(clf, 'model.joblib')