# XGBoost

from xgboost.sklearn import XGBClassifier

gb_clf = XGBClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0)
gb_clf.fit(x_train, y_train)

y_pred = gb_clf.predict(x_test)
round(accuracy_score(y_test,y_pred),3)
