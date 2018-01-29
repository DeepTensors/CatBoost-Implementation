import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import catboost

train = pd.read_csv('Train.csv')
test = pd.read_csv('test.csv')

train.dtypes

#Checking for null values
train.isnull().sum()

#Filling the na values
train.fillna(-999 , inplace = True)
test.fillna(-999,inplace=True)

X = train.drop(['Item_Outlet_Sales'] , axis =1)
y = train.Item_Outlet_Sales

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X, y, train_size=0.7,random_state=0)

categorical_features_indices = np.where(X.dtypes != np.float)[0]

model=catboost.CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test),plot=True)

submission = pd.DataFrame()
submission['Item_Identifier'] = test['Item_Identifier']
submission['Outlet_Identifier'] = test['Outlet_Identifier']
submission['Item_Outlet_Sales'] = model.predict(test)
submission.to_csv("Submission.csv")
