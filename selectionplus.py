import pandas as pd
import numpy as np
import shap
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import lightgbm as lgb
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

shap_values=0
shap_values_matrix=0
for _ in range(10):
    #noise = np.random.normal(size=xtrain.shape[0], loc=0, scale=0.05).reshape((-1, 1))
    lgb_train = lgb.Dataset(xtrain, ytrain)
    lgb_eval = lgb.Dataset(xval, yval, reference=lgb_train)
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              'objective': 'regression'}
    callback = [lgb.early_stopping(stopping_rounds=20, verbose=True),
                lgb.log_evaluation(period=20, show_stdv=True)]
    model = lgb.train(params, lgb_train, num_boost_round=1000,
                           valid_sets=[lgb_train, lgb_eval], callbacks=callback)

    #predict evaluate
    val_pred=model.predict(xval)
    val_pred=scalry.inverse_transform(val_pred.reshape((-1,1)))
    print("-------val set------------")
    print(r2_score(scalry.inverse_transform(yval),val_pred))
    print(mean_squared_error(scalry.inverse_transform(yval),val_pred))
    print("-------test set------------")
    test_pred=model.predict(xtest)
    test_pred=scalry.inverse_transform(test_pred.reshape((-1,1)))
    print(r2_score(scalry.inverse_transform(ytest),test_pred))
    print(mean_squared_error(scalry.inverse_transform(ytest),test_pred))

    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(xval)
    shap_values_matrix+=shap_value
    shap_value = np.mean(np.abs(shap_value), axis=0)
    shap_values+=shap_value
shap_values=shap_values/10
non_zero_id=np.where(shap_values>0)[0]

#plot the shap result
shap.summary_plot(np.array(shap_values_matrix),xval,feature_names=descriptor.columns)
shap.summary_plot(np.array(shap_values_matrix), xval, plot_type="bar",feature_names=descriptor.columns)
# create a SHAP dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot("nC", np.array(shap_values_matrix), xval,feature_names=descriptor.columns,interaction_index=None)
shap.dependence_plot("MDEC-23", np.array(shap_values_matrix), xval,feature_names=descriptor.columns,interaction_index=None)
shap.dependence_plot("maxHsOH", np.array(shap_values_matrix), xval,feature_names=descriptor.columns,interaction_index=None)
shap.dependence_plot("LipoaffinityIndex", np.array(shap_values_matrix), xval,feature_names=descriptor.columns,interaction_index=None)

#get the group score
group_score=[]
for g in group_list:
    group_score.append(np.mean(shap_values[g]))
plt.xlabel('group score')
plt.ylabel('group number')
plt.title('feature importance score of different group')
plt.barh(list(range(len(group_score))),group_score)
group_score=pd.DataFrame(group_score).to_csv('g_score1.csv')

import matplotlib.pyplot as plt
plt.scatter(list(range(len(shap_values))),shap_values*100)
plt.show()

non_zero_id=np.where(shap_values>0)[0]

variable_num=list(range(2,20,3))+list(range(20,len(non_zero_id),30))
times=10
mse_list=[]
r2_list=[]
for num in variable_num:
    for time in range(times):
        print("----------+1----------------------------")
        select_index_ = shap_values.argsort()[(-num)::]
        xtrain_, xval_=xtrain[:,select_index_],xval[:,select_index_]
        noise=np.random.normal(size=xtrain_.shape[0],loc=0,scale=0.05).reshape((-1,1))
        lgb_train = lgb.Dataset(xtrain_+noise, ytrain)
        lgb_eval = lgb.Dataset(xval_, yval, reference=lgb_train)
        params = {'task': 'train',
                  'boosting_type': 'gbdt',
                  'objective': 'regression'}
        callback = [lgb.early_stopping(stopping_rounds=20, verbose=True),
                    lgb.log_evaluation(period=20, show_stdv=True)]
        sub_model = lgb.train(params, lgb_train, num_boost_round=1000,
                          valid_sets=[lgb_train, lgb_eval], callbacks=callback)
        val_pred = sub_model.predict(xval_)
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
plt.savefig('importance+.png')


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
plt.savefig('importance2+.png')



select_index1 = shap_values.argsort()[(-50)::]
group_id=dict()
for i in select_index1:
    for index,g in enumerate(group_list):
        if i in g and index!=20:
            group_id[str(i)]=index
            break


select_index_ = shap_values.argsort()[(-50)::]
xtrain_, xval_=xtrain[:,select_index_],xval[:,select_index_]
noise=np.random.normal(size=xtrain_.shape[0],loc=0,scale=0.05).reshape((-1,1))
lgb_train = lgb.Dataset(xtrain_, ytrain)
lgb_eval = lgb.Dataset(xval_, yval, reference=lgb_train)
params = {'task': 'train',
          'boosting_type': 'gbdt',
          'objective': 'regression'}
callback = [lgb.early_stopping(stopping_rounds=20, verbose=True),
            lgb.log_evaluation(period=20, show_stdv=True)]
sub_model = lgb.train(params, lgb_train, num_boost_round=1000,
                  valid_sets=[lgb_train, lgb_eval], callbacks=callback)
val_pred = sub_model.predict(xval_)
val_pred = scalry.inverse_transform(val_pred.reshape((-1, 1)))
r2=r2_score(scalry.inverse_transform(yval), val_pred)
mse=mean_squared_error(scalry.inverse_transform(yval), val_pred)
print(r2)
sub_model.save_model('model_best50_2.json')

