from sklearn import svm
from sklearn import datasets
import joblib

# create model
#clf = svm.SVC(probability=True)
clf = xgb.XGBClassifier()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)

# save model
joblib.dump(clf, 'model.joblib')
pickle.dump(clf, open('model_pickle.pkl', 'wb'))