# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 22:36:13 2020

@author: ryan wang
"""



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset_Facebook.csv")
# 删除列
data = data.drop(['Page total likes','comment', 'like', 'share'], axis=1)

# 将time取对数
colNames = ['Lifetime Post Total Reach', 'Lifetime Post Total Impressions',
     'Lifetime Engaged Users', 'Lifetime Post Consumers','Lifetime Post Consumptions',
     'Lifetime Post Impressions by people who have liked your Page',
     'Lifetime Post reach by people who like your Page',
     'Lifetime People who have liked your Page and engaged with your post',
     'Total Interactions']
data[colNames] = np.log(data[colNames]+1)



# 描述性统计
for name in ['Type', 'Category', 'Post Month', 'Post Weekday', 'Post Hour', 'Paid']:
    summ = data[[name,'Total Interactions']].groupby(name).mean()
    summ.to_csv(f"tab\\{name}.csv")
    
# summ = data[["Type",'Total Interactions']].groupby("Type").mean()
# summ.to_csv("tab\\Type.csv")
# Type Category Paid对interaction影响比较明显
# Post Weekday Post Hour有一定的影响  Post Month影响不大
    

for name in colNames[:-1]:
    data[[name, 'Total Interactions']].plot(kind='scatter', x=name, y='Total Interactions')
    plt.savefig(f"fig\\{name}.jpeg")
    plt.close()



# 
newdata = []
for name in ['Type', 'Category', 'Post Month', 'Post Weekday', 'Post Hour', 'Paid']:
    dat = pd.get_dummies(data[name], drop_first=True, prefix=name)
    newdata.append(dat)
    
# dat1 = pd.get_dummies(data["Type"], drop_first=True)
# dat2 = pd.get_dummies(data["Category"], drop_first=True, prefix="Category")


for name in colNames[:-1]:
    dat = (data[[name]] - data[name].min()) / (data[name].max() - data[name].min())
    newdata.append(dat)

newdata.append(data[["Total Interactions"]])


data = pd.concat(newdata, axis=1)



# 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop(["Total Interactions"], axis=1), 
                                                    data["Total Interactions"], test_size=0.3, random_state=42)



# 线性回归
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
reg.coef_
reg.intercept_

# 训练误差 Root of mean square error
rmse_train_lm = np.sqrt(((reg.predict(X_train) - y_train)**2).mean())
rmse_test_lm = np.sqrt(((reg.predict(X_test) - y_test)**2).mean())




# 决策树
from sklearn.tree import DecisionTreeRegressor

para = np.arange(2, 150, 1)
train_err = []
test_err = []

for i in para:
    reg = DecisionTreeRegressor(min_samples_split=i, random_state=666)
    reg.fit(X_train, y_train)
    
    rmse_train_tree = np.sqrt(((reg.predict(X_train) - y_train)**2).mean())
    rmse_test_tree = np.sqrt(((reg.predict(X_test) - y_test)**2).mean())
    train_err.append(rmse_train_tree)
    test_err.append(rmse_test_tree)


err = pd.DataFrame([para, train_err, test_err], index=["Para", "TrainErr", "TestErr"]).T
err = err.set_index("Para")
err.plot(ylim=(0.4, 0.8))
plt.title("Train/Test Error of Decision Tree")
plt.savefig("fig\\tree.jpeg")
plt.close()


reg = DecisionTreeRegressor(min_samples_split=17, random_state=666)
reg.fit(X_train, y_train)
    
rmse_train_tree = np.sqrt(((reg.predict(X_train) - y_train)**2).mean())
rmse_test_tree = np.sqrt(((reg.predict(X_test) - y_test)**2).mean())



# 

from sklearn.ensemble import RandomForestRegressor

para = np.arange(1, 300, 5)
train_err = []
test_err = []

for i in para:
    print(i)
    reg = RandomForestRegressor(n_estimators=i, random_state=666)
    reg.fit(X_train, y_train)
    
    rmse_train_tree = np.sqrt(((reg.predict(X_train) - y_train)**2).mean())
    rmse_test_tree = np.sqrt(((reg.predict(X_test) - y_test)**2).mean())
    train_err.append(rmse_train_tree)
    test_err.append(rmse_test_tree)


err = pd.DataFrame([para, train_err, test_err], index=["Para", "TrainErr", "TestErr"]).T
err = err.set_index("Para")
err.plot()
plt.title("Train/Test Error of Random Forest")
plt.savefig("fig\\rf.jpeg")
plt.close()


reg = RandomForestRegressor(n_estimators=76, random_state=666)
reg.fit(X_train, y_train)
    
rmse_train_rf = np.sqrt(((reg.predict(X_train) - y_train)**2).mean())
rmse_test_rf = np.sqrt(((reg.predict(X_test) - y_test)**2).mean())







from sklearn.ensemble import AdaBoostRegressor

para = np.arange(1, 600, 10)
train_err = []
test_err = []

for i in para:
    print(i)
    reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=10), n_estimators=i, random_state=666)
    reg.fit(X_train, y_train)
    
    rmse_train_tree = np.sqrt(((reg.predict(X_train) - y_train)**2).mean())
    rmse_test_tree = np.sqrt(((reg.predict(X_test) - y_test)**2).mean())
    train_err.append(rmse_train_tree)
    test_err.append(rmse_test_tree)


err = pd.DataFrame([para, train_err, test_err], index=["Para", "TrainErr", "TestErr"]).T
err = err.set_index("Para")
err.plot()
plt.title("Train/Test Error of AdaBoostRegressor")
plt.savefig("fig\\bst.jpeg")
plt.close()


reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=10), n_estimators=361, random_state=666)
reg.fit(X_train, y_train)
    
rmse_train_bst = np.sqrt(((reg.predict(X_train) - y_train)**2).mean())
rmse_test_bst = np.sqrt(((reg.predict(X_test) - y_test)**2).mean())




# 
from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(hidden_layer_sizes=(600,), random_state=1, max_iter=1000, 
                   verbose=True, activation = 'logistic', learning_rate='constant', learning_rate_init=0.001)
reg.fit(X_train, y_train)
    
rmse_train_nn = np.sqrt(((reg.predict(X_train) - y_train)**2).mean())
rmse_test_nn = np.sqrt(((reg.predict(X_test) - y_test)**2).mean())
rmse_train_nn
rmse_test_nn







train_err = [rmse_train_lm, rmse_train_tree, rmse_train_rf, rmse_train_bst, rmse_train_nn]
test_err = [rmse_test_lm, rmse_test_tree, rmse_test_rf, rmse_test_bst, rmse_test_nn]
summ = pd.DataFrame([train_err, test_err], index=["TrainErr", "TestErr"], 
                    columns = ["Linear Regression", "Tree", "Random Forest", "AdaBoosting", "Neural Network"]).T
summ.to_csv("summ.csv")
