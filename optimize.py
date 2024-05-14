import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import geatpy as ea
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBClassifier
from entropy_estimators import continuous
# test the loaded model
select_index_=[238, 723,  53, 622, 651, 346, 589, 475, 273, 611,  38, 470, 237,
       652,  41,  23, 105, 465, 665,  22, 508,  79, 648, 531,  70, 406,
       103,  40, 412, 642,   1, 351,  83, 661, 154, 727, 716, 356,  42,
       410,  58, 476,  39, 673, 639, 357,  56, 587,  11, 659]



#genetic algorithm class
solution_num=len(select_index_)
class MyProblem(ea.Problem):
    def __init__(self,model,clf1,clf2,clf3,clf4,clf5):
        name = 'MyProblem'  # initialize name
        M = 1  # objective dimensions
        maxormins = [-1]
        Dim = solution_num
        varTypes = [0] * Dim  # varTypes
        lb = [0]*Dim  # lower bound
        ub = [1]*Dim  # upper bound
        lbin = [1] * Dim
        ubin = [1] * Dim
        self.model=model
        self.clf1 = clf1
        self.clf2 = clf2
        self.clf4 = clf4
        self.clf3 = clf3
        self.clf5 = clf5
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  #objective function
        #solution variable matrix,each row represents a probably sample
        Vars = pop.Phen
        #get the Ealpha activity
        activity_pred=self.model.predict(xgboost.DMatrix(Vars))
        current_obj=scalry.inverse_transform(activity_pred.reshape((-1, 1)))
        pop.ObjV =current_obj
        #get the admet
        alpha1,alpha2,alpha3,alpha4,alpha5=self.clf1.predict(Vars).reshape((-1, 1)),self.clf2.predict(Vars).reshape((-1, 1)),self.clf3.predict(Vars).reshape((-1, 1)),self.clf4.predict(Vars).reshape((-1, 1)),self.clf5.predict(Vars).reshape((-1, 1))
        pop.CV = np.hstack([-alpha1-alpha2-alpha3-alpha4-alpha5+3])
    def calReferObjV(self):
        referenceObjV = np.array([[10.33]])
        return referenceObjV
if __name__=="__main__":
    # load model
    model_best50 = xgboost.Booster()
    model_best50.load_model("model_best50.json")
    admet = pd.read_excel('ADMET.xlsx')

    clf1 = XGBClassifier()
    booster = xgboost.Booster()
    booster.load_model('clf1.json')
    clf1._Booster = booster

    clf2 = XGBClassifier()
    booster = xgboost.Booster()
    booster.load_model('clf2.json')
    clf2._Booster = booster

    clf3 = XGBClassifier()
    booster = xgboost.Booster()
    booster.load_model('clf3.json')
    clf3._Booster = booster

    clf4 = XGBClassifier()
    booster = xgboost.Booster()
    booster.load_model('clf4.json')
    clf4._Booster = booster

    clf5 = XGBClassifier()
    booster = xgboost.Booster()
    booster.load_model('clf5.json')
    clf5._Booster = booster


    # load the data
    descriptor = pd.read_excel('Molecular_Descriptor.xlsx')
    activity = pd.read_excel('ERα_activity.xlsx')
    data = pd.merge(left=descriptor, right=activity, on='SMILES')

    x, y = data.iloc[:, 1:-2], data.iloc[:, -1]
    x, y = np.array(x), np.array(y)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15, random_state=1)
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.1, random_state=1)

    #compare mutual information
    residul_index=[ i for i in list(range(x.shape[1])) if i not in select_index_ ]
    in_50,not_in_50=[],[]
    index=np.random.permutation(len(residul_index))[:50]
    for i,j in zip(residul_index[-50::],select_index_):
        x_i,x_j=x[:,i],x[:,j]
        in_50.append(continuous.get_mi(x_j,y))
        not_in_50.append(continuous.get_mi(x_i,y))

    np.mean(not_in_50)
    np.mean(in_50)
    res=np.array(in_50)-np.array(not_in_50)

    # normalize
    scalrx, scalry = MinMaxScaler(), MinMaxScaler()
    xtrain, ytrain = scalrx.fit_transform(xtrain), scalry.fit_transform(ytrain.reshape((-1, 1)))
    xval, xtest, yval, ytest = scalrx.transform(xval), scalrx.transform(xtest), scalry.transform(
        yval.reshape((-1, 1))), scalry.transform(ytest.reshape((-1, 1)))

    xval_, xtest_, xtrain_ = xval[:, select_index_], xtest[:, select_index_], xtrain[:, select_index_]
    val_pred = model_best50.predict(xgboost.DMatrix(xval_))
    val_pred = scalry.inverse_transform(val_pred.reshape((-1, 1)))
    r2 = r2_score(scalry.inverse_transform(yval), val_pred)
    mse = mean_squared_error(scalry.inverse_transform(yval), val_pred)
    print(r2)





    #optimize
    k=200
    _k_sort = np.argpartition(y, -k)[-k:]
    spare_variable_array=scalrx.transform(x[_k_sort,:])[:,select_index_]


    problem = MyProblem(model_best50,clf1,clf2,clf3,clf4,clf5)
    """==================================population setting=============================="""
    Encoding = 'RI'  # encoding method
    NIND = k  #  population size
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND,Chrom=spare_variable_array)
    """================================parameters setting============================="""
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population)
    myAlgorithm.MAXGEN = 5000  # max evolution generation
    myAlgorithm.mutOper.F = 0.5  #
    myAlgorithm.recOper.XOVR = 0.8  # mutation
    myAlgorithm.logTras = 1  #
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 1  #
    """===========================run========================"""
    [BestIndi, population] = myAlgorithm.run()
    BestIndi.save()
    """=================================result=============================="""
    print('times：%s' % myAlgorithm.evalsNum)
    print('time elapse %s second' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('objective values：%s' % BestIndi.ObjV[0][0])
        print('best solution num：')
        for i in range(BestIndi.Phen.shape[1]):
            print(BestIndi.Phen[0, i])
    else:
        print('no solution')