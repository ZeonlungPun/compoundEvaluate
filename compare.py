import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from lightgbm.sklearn import LGBMRegressor
import seaborn as sns
import matplotlib.pyplot as plt
# load the data
descriptor=pd.read_csv('descriptor_.csv')
activity=pd.read_excel('ERα_activity.xlsx')
shap_value=pd.read_csv('shap.csv').loc[:,'shap']
non_zero_id=np.where(shap_value>0)[0]
x,y=descriptor,activity.loc[:,"pIC50"]
x,y=np.array(x),np.array(y)
# test the loaded model
select_index_=[641,465, 646,  39,  99, 474, 102,  20, 469, 528,
       290,  22, 386, 722, 232,  69, 647,  40, 464,  37, 663, 621,  41,
       660, 409, 725,   1, 153,  82, 411, 530, 715, 355, 405,  57, 664,
       350, 651, 356,  38, 672, 638,  55,  10, 475, 586, 658,98,613,701]

resiual_id=np.array([ i for i in list(range(len(shap_value))) if i not in non_zero_id])
next50_id=list(shap_value.argsort()[(-100):(-50)])
next100_id=list(shap_value.argsort()[(-150):(-100)])
randex_index=np.random.permutation(len(resiual_id))
random1_id=resiual_id[randex_index[0:50]]
random2_id=randex_index[50:100]
result1,result2,result3,result4,result5=[],[],[],[],[]
for _ in range(20):
       xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)
       xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.1)
       xval_1, xtest_1, xtrain_1 = xval[:, select_index_], xtest[:, select_index_], xtrain[:, select_index_]
       xval_2, xtest_2, xtrain_2 = xval[:, next50_id], xtest[:, next50_id], xtrain[:, next50_id]
       xval_3, xtest_3, xtrain_3 = xval[:, next100_id], xtest[:, next100_id], xtrain[:, next100_id]
       xval_4, xtest_4, xtrain_4 = xval[:, random1_id], xtest[:, random1_id], xtrain[:, random1_id]
       xval_5, xtest_5, xtrain_5 = xval[:, random2_id], xtest[:, random2_id], xtrain[:, random2_id]
       # normalize
       scalrx1 = MinMaxScaler()
       xtrain1 = scalrx1.fit_transform(xtrain_1)
       xval1, xtest1 = scalrx1.transform(xval_1), scalrx1.transform(xtest_1)

       scalrx2 = MinMaxScaler()
       xtrain2 = scalrx2.fit_transform(xtrain_2)
       xval2, xtest2 = scalrx2.transform(xval_2), scalrx2.transform(xtest_2)

       scalrx3 = MinMaxScaler()
       xtrain3 = scalrx3.fit_transform(xtrain_3)
       xval3, xtest3 = scalrx3.transform(xval_3), scalrx3.transform(xtest_3)

       scalrx4 = MinMaxScaler()
       xtrain4 = scalrx4.fit_transform(xtrain_4)
       xval4, xtest4 = scalrx4.transform(xval_4), scalrx4.transform(xtest_4)

       scalrx5 = MinMaxScaler()
       xtrain5 = scalrx5.fit_transform(xtrain_5)
       xval5, xtest5 = scalrx5.transform(xval_5), scalrx5.transform(xtest_5)
       
       lgb1=LGBMRegressor()
       lgb2 = LGBMRegressor()
       lgb3 = LGBMRegressor()
       lgb4 = LGBMRegressor()
       lgb5 = LGBMRegressor()
       lgb1.fit(xtrain1,ytrain)
       lgb2.fit(xtrain2,ytrain)
       lgb3.fit(xtrain3,ytrain)
       lgb4.fit(xtrain4, ytrain)
       lgb5.fit(xtrain5, ytrain)
       pred1,pred2,pred3,pred4,pred5=lgb1.predict(xval1),lgb2.predict(xval2),lgb3.predict(xval3),lgb4.predict(xval4),lgb5.predict(xval5)
       acc1,acc2,acc3,acc4,acc5=r2_score(yval,pred1),r2_score(yval,pred2),r2_score(yval,pred3),r2_score(yval,pred4),r2_score(yval,pred5)
       result1.append(acc1)
       result2.append(acc2)
       result3.append(acc3)
       result4.append(acc4)
       result5.append(acc5)

total = pd.DataFrame({'top 50': result1, 'rank:50-100': result2,'rank:100-150':result3,'not important':result4,'random':result5})
#total.boxplot()
sns.boxplot(data=total)
#plt.xticks(ticks=np.arange(0, len(total.columns), labels=total.columns), rotation=45)
plt.ylabel('R-square')
plt.title('performance of LightGBM model using different descriptor')
plt.show()

#senstive analysis
respond1=descriptor.loc[:,'MDEC-23']
plt.scatter(np.array(respond1),y)
plt.ylabel('pIC50')
plt.xlabel('MDEC-23')
plt.show()

respond2=descriptor.loc[:,'LipoaffinityIndex']
plt.scatter(np.array(respond2),y)
plt.ylabel('pIC50')
plt.xlabel('LipoaffinityIndex')
plt.show()


respond3=descriptor.loc[:,'nC']
plt.scatter(np.array(respond3),y)
plt.ylabel('pIC50')
plt.xlabel('nC')
plt.show()


respond4=descriptor.loc[:,'maxHsOH']
plt.scatter(np.array(respond4),y)
plt.ylabel('pIC50')
plt.xlabel('maxHsOH')
plt.show()