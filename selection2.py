import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
import seaborn as sns
# load the data
descriptor=pd.read_csv('descriptor_.csv')
admet = pd.read_excel('ADMET.xlsx')


x,y=descriptor,admet.iloc[:,1::]
x,y=np.array(x),np.array(y)
# test the loaded model
select_index_=[224, 641, 391, 465, 646,  39,  99, 588, 474, 102,  20, 469, 528,
       290,  22, 386, 722, 232,  69, 647,  40, 464,  37, 663, 621,  41,
       660, 409, 725,   1, 153,  82, 411, 530, 715, 355, 405,  57, 664,
       350, 651, 356,  38, 672, 638,  55,  10, 475, 586, 658]




acc_list1,acc_list2,acc_list3,acc_list4,acc_list5=[],[],[],[],[]
#multi-output
for _ in range(10):
       xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)
       xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.1)
       xval_, xtest_, xtrain_ = xval[:, select_index_], xtest[:, select_index_], xtrain[:, select_index_]
       # normalize
       scalrx = MinMaxScaler()
       xtrain = scalrx.fit_transform(xtrain_)
       xval, xtest = scalrx.transform(xval_), scalrx.transform(xtest_)
       clf = XGBClassifier()
       clf.fit(xtrain, ytrain)
       val_pred=clf.predict(xval)
       acc1=accuracy_score(yval[:,0],val_pred[:,0])
       acc2=accuracy_score(yval[:,1],val_pred[:,1])
       acc3=accuracy_score(yval[:,2],val_pred[:,2])
       acc4=accuracy_score(yval[:,3],val_pred[:,3])
       acc5=accuracy_score(yval[:,4],val_pred[:,4])
       acc_list1.append(acc1)
       acc_list2.append(acc2)
       acc_list3.append(acc3)
       acc_list4.append(acc4)
       acc_list5.append(acc5)

#single output model
acc_list1_,acc_list2_,acc_list3_,acc_list4_,acc_list5_=[],[],[],[],[]



for _ in range(10):
       xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)
       xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.1)
       xval_, xtest_, xtrain_ = xval[:, select_index_], xtest[:, select_index_], xtrain[:, select_index_]
       # normalize
       scalrx = MinMaxScaler()
       xtrain = scalrx.fit_transform(xtrain_)
       xval, xtest = scalrx.transform(xval_), scalrx.transform(xtest_)

       ytrain1, ytest1, yval1 = ytrain[:, 0], ytest[:, 0], yval[:, 0]
       ytrain2, ytest2, yval2 = ytrain[:, 1], ytest[:, 1], yval[:, 1]
       ytrain3, ytest3, yval3 = ytrain[:, 2], ytest[:, 2], yval[:, 2]
       ytrain4, ytest4, yval4 = ytrain[:, 3], ytest[:, 3], yval[:, 3]
       ytrain5, ytest5, yval5 = ytrain[:, 4], ytest[:, 4], yval[:, 4]
       clf1 = LGBMClassifier()
       clf1.fit(xtrain, ytrain1)
       val_pred=clf1.predict(xval)
       acc1=accuracy_score(yval1.reshape((-1,1)),val_pred.reshape((-1,1)))

       clf2 =LGBMClassifier()
       clf2.fit(xtrain, ytrain2)
       val_pred=clf2.predict(xval)
       acc2=accuracy_score(yval2.reshape((-1,1)),val_pred.reshape((-1,1)))

       clf3 = LGBMClassifier()
       clf3.fit(xtrain, ytrain3)
       val_pred=clf3.predict(xval)
       acc3=accuracy_score(yval3.reshape((-1,1)),val_pred.reshape((-1,1)))

       clf4 = LGBMClassifier()
       clf4.fit(xtrain, ytrain4)
       val_pred=clf4.predict(xval)
       acc4=accuracy_score(yval4.reshape((-1,1)),val_pred.reshape((-1,1)))

       clf5 = LGBMClassifier()
       clf5.fit(xtrain, ytrain5)
       val_pred=clf5.predict(xval)
       acc5=accuracy_score(yval5.reshape((-1,1)),val_pred.reshape((-1,1)))
       acc_list1_.append(acc1)
       acc_list2_.append(acc2)
       acc_list3_.append(acc3)
       acc_list4_.append(acc4)
       acc_list5_.append(acc5)
print(np.mean(acc_list1),np.mean(acc_list2),np.mean(acc_list3),np.mean(acc_list4),np.mean(acc_list5))
print(np.std(acc_list1),np.std(acc_list2),np.std(acc_list3),np.std(acc_list4),np.std(acc_list5))
print(np.mean(acc_list1_), np.mean(acc_list2_), np.mean(acc_list3_), np.mean(acc_list4_), np.mean(acc_list5_))
print(np.std(acc_list1_),np.std(acc_list2_),np.std(acc_list3_),np.std(acc_list4_),np.std(acc_list5_))

# clf1.save_model('clf1.json')
# clf2.save_model('clf2.json')
# clf3.save_model('clf3.json')
# clf4.save_model('clf4.json')
# clf5.save_model('clf5.json')