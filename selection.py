import pandas as pd
import numpy as np
import shap,xgboost
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#load the data
descriptor=pd.read_csv('descriptor_.csv')
describe_=pd.read_csv('describe_.csv')
activity=pd.read_excel('ERα_activity.xlsx')
group=describe_.loc[:,'group']
group_num=pd.unique(group)
group_list=[]
for i in group_num:
    feature_id=list(np.where(group==i)[0])
    group_list.append(feature_id)
x=descriptor
y=activity.loc[:,"pIC50"]

x, y = np.array(x), np.array(y)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.15,random_state=1)
xtrain,xval,ytrain,yval=train_test_split(xtrain,ytrain,test_size=0.1,random_state=1)

#normalize
scalrx,scalry=MinMaxScaler(),MinMaxScaler()
xtrain,ytrain=scalrx.fit_transform(xtrain),scalry.fit_transform(ytrain.reshape((-1,1)))
xval,xtest,yval,ytest=scalrx.transform(xval),scalrx.transform(xtest),scalry.transform(yval.reshape((-1,1))),scalry.transform(ytest.reshape((-1,1)))



# train the model
model = xgboost.train({"learning_rate": 0.01,"subsample": 0.65}, xgboost.DMatrix(xtrain, label=ytrain), 500)


#predict evaluate
val_pred=model.predict(xgboost.DMatrix(xval))
val_pred=scalry.inverse_transform(val_pred.reshape((-1,1)))
print("-------val set------------")
print(r2_score(scalry.inverse_transform(yval),val_pred))
print(mean_squared_error(scalry.inverse_transform(yval),val_pred))
print("-------test set------------")
test_pred=model.predict(xgboost.DMatrix(xtest))
test_pred=scalry.inverse_transform(test_pred.reshape((-1,1)))
print(r2_score(scalry.inverse_transform(ytest),test_pred))
print(mean_squared_error(scalry.inverse_transform(ytest),test_pred))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(xval)
shap_values = np.mean(np.abs(shap_values), axis=0)
non_zero_id=np.where(shap_values>0)[0]

variable_num=list(range(2,20,3))+list(range(20,len(non_zero_id),30))
times=5
mse_list=[]
r2_list=[]
for num in variable_num:
    for time in range(times):
        print("----------+1----------------------------")
        select_index_ = shap_values.argsort()[(-num)::]
        xtrain_, xval_=xtrain[:,select_index_],xval[:,select_index_]
        noise=np.random.normal(size=xtrain_.shape[0],loc=0,scale=0.05).reshape((-1,1))
        sub_model = xgboost.train({"learning_rate": 0.01,"subsample": 0.65}, xgboost.DMatrix(xtrain_+noise, label=ytrain), 500)
        val_pred = sub_model.predict(xgboost.DMatrix(xval_))
        val_pred = scalry.inverse_transform(val_pred.reshape((-1, 1)))
        r2=r2_score(scalry.inverse_transform(yval), val_pred)
        mse=mean_squared_error(scalry.inverse_transform(yval), val_pred)
        mse_list.append(mse)
        r2_list.append(r2)


#plot the boxplot
ind = [i for i in variable_num for _ in range(times)]
df = pd.DataFrame({'index': ind, 'mse': mse_list})
fig, ax = plt.subplots()
# Boxplot
df.boxplot(column='mse', by='index', ax=ax, grid=False)

# Median line and points
medians = df.groupby('index')['mse'].median()
ax.plot(range(1, len(variable_num)+1), medians, color='red', linestyle='--', label='Median')
ax.scatter(range(1, len(variable_num)+1), medians, color='red', marker='o')
ax.set_xlabel('the most important features used')
ax.set_ylabel('MSE')
ax.set_title('Prediction error trend by important features used ')
plt.legend()
plt.savefig('importance.png')


df2 = pd.DataFrame({'index': ind, 'r2': r2_list})
fig, ax = plt.subplots()
# Boxplot
df2.boxplot(column='r2', by='index', ax=ax, grid=False)

# Median line and points
medians = df2.groupby('index')['r2'].median()
ax.plot(range(1, len(variable_num)+1), medians, color='red', linestyle='--', label='Median')
ax.scatter(range(1, len(variable_num)+1), medians, color='red', marker='o')
ax.set_xlabel('the most important features used')
ax.set_ylabel('r2')
ax.set_title('Prediction accuracy trend by important features used ')
plt.legend()
plt.savefig('importance2.png')

#save the model
select_index_ = shap_values.argsort()[(-50)::]
xtrain_, xval_=xtrain[:,select_index_],xval[:,select_index_]
#noise=np.random.normal(size=xtrain_.shape[0],loc=0,scale=0.05).reshape((-1,1))
sub_model = xgboost.train({"learning_rate": 0.01,"subsample": 0.65}, xgboost.DMatrix(xtrain_, label=ytrain), 500)
val_pred = sub_model.predict(xgboost.DMatrix(xval_))
val_pred = scalry.inverse_transform(val_pred.reshape((-1, 1)))
r2=r2_score(scalry.inverse_transform(yval), val_pred)
mse=mean_squared_error(scalry.inverse_transform(yval), val_pred)
print(r2)
sub_model.save_model('model_best50.json')