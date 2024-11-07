import pandas as pd
import numpy as np


data1=pd.read_csv("soybean_marker_selection.csv")
data1=data1.iloc[:,0:3]

data2=pd.read_csv('flower_need.csv')
data2['snp'] = data2.iloc[:,0].str.split('_').str[0]
data2.index=data2['snp']
data1.index=data1['snp']
data=pd.merge(data1, data2, left_index=True, right_index=True, how='inner')
new_data=data.iloc[:,[1,2,4,5,6]]
new_data.to_csv('data_flower.csv')

import matplotlib.pyplot as plt
import seaborn as sns
#for plotting
lgb=np.array([0.9002240857267731, 0.883220022767118, 0.8739155837169853, 0.8919995971640857, 0.8686765697484231, 0.8895108775378842, 0.8806779252027646, 0.8711405258974011, 0.8605073656130194, 0.8916268025392871])
xgb=np.array([0.849489548302048, 0.8738014066863153, 0.8406240051683617, 0.8644377064412118, 0.8567903106056881, 0.8682156648457885, 0.873015206198184, 0.8421734071053104, 0.8421757052667265, 0.8587330897923531])
svm=np.array([0.8484791351705805, 0.8272014334938633, 0.8119164866173328, 0.8603605855386574, 0.8591969845462136, 0.8350655991867176, 0.8567836641222114, 0.8317862672105288, 0.8593064238690309, 0.8416522789496212])
mlp=np.array([0.7938011259933505, 0.8080625980829529, 0.8104583624205067, 0.7757879084100376, 0.8069045317828774, 0.7866264370332923, 0.7884689567632898, 0.7791036989830106, 0.8083932779858455, 0.8243390930557443])
rf=np.array([0.8893535533834515, 0.8621748366161462, 0.8772250330711193, 0.8592560313794213, 0.8795297309880766, 0.8603379428594379, 0.8904786567305868, 0.8653560534371094, 0.868476025064964, 0.8724754655852708])
total = pd.DataFrame({'RF': rf**2, 'XGB': xgb**2,'LGB':lgb**2 ,'SVM': svm**2, 'MLP': mlp**2})
#total.boxplot()
sns.violinplot(data=total)
#plt.xticks(ticks=np.arange(0, len(total.columns), labels=total.columns), rotation=45)
plt.ylabel('R-square')
plt.title('performance of different models')
plt.show()


# 创建DataFrame
y1 = pd.DataFrame(acc_list1)
y1_ = pd.DataFrame(acc_list1_)

y2 = pd.DataFrame(acc_list2)
y2_ = pd.DataFrame(acc_list2_)

y3 = pd.DataFrame(acc_list3)
y3_ = pd.DataFrame(acc_list3_)

y4 = pd.DataFrame(acc_list4)
y4_ = pd.DataFrame(acc_list4_)

y5 = pd.DataFrame(acc_list5)
y5_ = pd.DataFrame(acc_list5_)

# 合并DataFrame
g1 = pd.concat([y1, y2, y3, y4, y5], axis=1)
g2 = pd.concat([y1_, y2_, y3_, y4_, y5_], axis=1)
g1,g2=np.array(g1),np.array(g2)

# 确保positions长度与g1和g2列数一致

x1 = np.arange(1,18,4)
x2 = x1+1

# 绘制箱线图
box1 = plt.boxplot(g1, positions=x1, patch_artist=True, showmeans=True,
                   boxprops={"facecolor": "C0", "edgecolor": "grey", "linewidth": 0.5},
                   medianprops={"color": "k", "linewidth": 0.5},
                   meanprops={'marker': '+', 'markerfacecolor': 'k', 'markeredgecolor': 'k', 'markersize': 5})

box2 = plt.boxplot(g2, positions=x2, patch_artist=True, showmeans=True,
                   boxprops={"facecolor": "C1", "edgecolor": "grey", "linewidth": 0.5},
                   medianprops={"color": "k", "linewidth": 0.5},
                   meanprops={'marker': '+', 'markerfacecolor': 'k', 'markeredgecolor': 'k', 'markersize': 5})

city = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
plt.xticks((x1 + x2) / 2, city, fontsize=9.5)
plt.ylabel('person correlation', fontsize=11)
plt.grid(axis='y', ls='--', alpha=0.8)

plt.legend(handles=[box1['boxes'][0], box2['boxes'][0]], labels=['multi-output model', 'single-output model'])
plt.title('results comparison between multi-output and single-output model ')
plt.show()

from scipy import stats
statistic, p_value = stats.mannwhitneyu(acc_list5, acc_list5_, alternative='two-sided')
print("U統計量:", statistic)
print("p值:", p_value)

svm1=[0.8690476190476191,
 0.8452380952380952,
 0.8333333333333334,
 0.8571428571428571,
 0.8809523809523809,
 0.8869047619047619,
 0.8690476190476191,
 0.8333333333333334,
 0.8511904761904762,
 0.8690476190476191]

svm2=[0.8988095238095238,
 0.8809523809523809,
 0.8809523809523809,
 0.8869047619047619,
 0.9107142857142857,
 0.9047619047619048,
 0.9404761904761905,
 0.9107142857142857,
 0.9047619047619048,
 0.9583333333333334]

svm3=[0.875,
 0.8630952380952381,
 0.8690476190476191,
 0.8333333333333334,
 0.875,
 0.8928571428571429,
 0.8511904761904762,
 0.8571428571428571,
 0.8392857142857143,
 0.8988095238095238]

svm4=[0.7916666666666666,
 0.8630952380952381,
 0.8392857142857143,
 0.8154761904761905,
 0.75,
 0.8095238095238095,
 0.8154761904761905,
 0.8214285714285714,
 0.7440476190476191,
 0.8035714285714286]

svm5=[0.875,
 0.9107142857142857,
 0.8630952380952381,
 0.8809523809523809,
 0.8809523809523809,
 0.8809523809523809,
 0.8869047619047619,
 0.8511904761904762,
 0.875,
 0.8928571428571429]






