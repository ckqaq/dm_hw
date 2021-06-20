## 决策树（decision tree）

决策树算法起源于E.B.Hunt等人于1966年发表的论文“experiments in Induction”，但真正让决策树成为机器学习主流算法的还是Quinlan（罗斯.昆兰）（2011年获得了数据挖掘领域最高奖KDD创新奖），昆兰在1979年提出了ID3算法，掀起了决策树研究的高潮。现在最常用的决策树算法是C4.5是昆兰在1993年提出的。

顾名思义，决策树就是一棵树，一颗决策树包含一个根节点、若干个内部结点和若干个叶结点；叶结点对应于决策结果，其他每个结点则对应于一个属性测试；每个结点包含的样本集合根据属性测试的结果被划分到子结点中；根结点包含样本全集，从根结点到每个叶子结点的路径对应了一个判定测试序列。下面直接上个图，让大家看下决策树是怎样决策的（以二元分类为例），图中红线表示给定一个样例（表中数据）决策树的决策过程：




![1](\pic\1.png)





## 梯度提升(GBM)



梯度提升是一种用于回归和分类问题的机器学习技术，该技术以弱预测模型(通常为决策树)的集合的形式产生预测模型。

任何监督学习算法的目标是定义一个损失函数并将其最小化。让我们看看梯度提升算法的数学运算。假设我们将均方误差(MSE)定义为：

![2](\pic\2.jpg)



我们希望我们的预测，使我们的损失函数(MSE)最小。通过使用**梯度下降**并根据学习速率更新我们的预测，我们可以找到MSE最小的值。

![3](\pic\3.jpg)

因此，我们更新预测使得我们的残差总和接近于0(或最小值)，并且预测值足够接近实际值。

#### **梯度提升的直观理解**

梯度提升背后的逻辑很简单，(可以直观地理解，不使用数学符号)。我希望阅读这篇文章的人可能会很熟悉`simple linear regression`模型。

线性回归的一个基本假设是其残差之和为0，即残差应该在零附近随机扩散。

![4](\pic\4.jpg)



现在将这些残差视为我们的预测模型犯下的错误。虽然，树模型*(考虑决策树作为我们梯度提升的基础模型)*不是基于这样的假设，但如果我们从逻辑上(而不是统计上)考虑这个假设

所以，背后的直觉`梯度提升（gradient boosting）`算法是重复利用残差中的模式，并加强一个弱预测模型并使其更好。一旦我们达到残差没有任何可模拟模式的阶段，我们可以停止建模残差(否则可能导致过拟合)。在算法上，持续最小化损失函数，使得测试损失达到最小值。

> 综上所述，
> •我们首先用简单的模型对数据进行建模并分析数据中的错误。
> •这些错误表示难以用简单模型拟合的数据点。
> •然后对于以后的模型，我们特别关注那些难以拟合的数据点，以使他们正确。
> •最后，我们通过给每个预测变量赋予一些权重来组合所有预测变量。



## 对决策树方法和GBM方法的实现



### 数据预处理

主要是对训练集进行预处理 去掉Label列 之后对训练集进行随机分配2:8比例 测试方法准确度


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
```


```python
train = pd.read_csv('C:/Users/15192/Desktop/pro_train(1).csv')
```


```python
train.shape
```


    (5227, 23)


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5227 entries, 0 to 5226
    Data columns (total 23 columns):
     #   Column                                 Non-Null Count  Dtype  
    ---  ------                                 --------------  -----  
     0   ID                                     5227 non-null   int64  
     1   Dependents                             5227 non-null   int64  
     2   MonthlyCharges                         5227 non-null   float64
     3   Partner                                5227 non-null   int64  
     4   PhoneService                           5227 non-null   int64  
     5   SeniorCitizen                          5227 non-null   int64  
     6   TotalCharges                           5227 non-null   float64
     7   tenure                                 5227 non-null   int64  
     8   Label                                  5227 non-null   int64  
     9   Contract_One year                      5227 non-null   int64  
     10  Contract_Two year                      5227 non-null   int64  
     11  InternetService_Fiber optic            5227 non-null   int64  
     12  InternetService_No                     5227 non-null   int64  
     13  PaymentMethod_Credit card (automatic)  5227 non-null   int64  
     14  PaymentMethod_Electronic check         5227 non-null   int64  
     15  PaymentMethod_Mailed check             5227 non-null   int64  
     16  gender_Male                            5227 non-null   int64  
     17  DeviceProtection_No                    5227 non-null   int64  
     18  DeviceProtection_Yes                   5227 non-null   int64  
     19  MultipleLines_No                       5227 non-null   int64  
     20  MultipleLines_Yes                      5227 non-null   int64  
     21  TVProgram_No                           5227 non-null   int64  
     22  TVProgram_Yes                          5227 non-null   int64  
    dtypes: float64(2), int64(21)
    memory usage: 939.4 KB

```python
train.describe(
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Dependents</th>
      <th>MonthlyCharges</th>
      <th>Partner</th>
      <th>PhoneService</th>
      <th>SeniorCitizen</th>
      <th>TotalCharges</th>
      <th>tenure</th>
      <th>Label</th>
      <th>Contract_One year</th>
      <th>...</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
      <th>gender_Male</th>
      <th>DeviceProtection_No</th>
      <th>DeviceProtection_Yes</th>
      <th>MultipleLines_No</th>
      <th>MultipleLines_Yes</th>
      <th>TVProgram_No</th>
      <th>TVProgram_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>...</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
      <td>5227.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2613.000000</td>
      <td>0.225368</td>
      <td>66.823765</td>
      <td>0.423379</td>
      <td>0.929214</td>
      <td>0.118615</td>
      <td>2084.477153</td>
      <td>28.775971</td>
      <td>0.372489</td>
      <td>0.165870</td>
      <td>...</td>
      <td>0.163382</td>
      <td>0.481538</td>
      <td>0.188827</td>
      <td>0.493017</td>
      <td>0.531280</td>
      <td>0.287737</td>
      <td>0.486321</td>
      <td>0.442893</td>
      <td>0.438492</td>
      <td>0.380524</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1509.049259</td>
      <td>0.417865</td>
      <td>28.862749</td>
      <td>0.494142</td>
      <td>0.256492</td>
      <td>0.323366</td>
      <td>2183.825066</td>
      <td>24.293077</td>
      <td>0.483514</td>
      <td>0.371999</td>
      <td>...</td>
      <td>0.369750</td>
      <td>0.499707</td>
      <td>0.391409</td>
      <td>0.499999</td>
      <td>0.499068</td>
      <td>0.452751</td>
      <td>0.499861</td>
      <td>0.496776</td>
      <td>0.496250</td>
      <td>0.485562</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1306.500000</td>
      <td>0.000000</td>
      <td>45.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>292.979609</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2613.000000</td>
      <td>0.000000</td>
      <td>74.200000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1218.650000</td>
      <td>23.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3919.500000</td>
      <td>0.000000</td>
      <td>89.900000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3373.825000</td>
      <td>51.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5226.000000</td>
      <td>1.000000</td>
      <td>118.600000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>8564.750000</td>
      <td>72.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 23 columns</p>
</div>


```python
y = train.Label
```


```python
y
```


    0       0
    1       1
    2       0
    3       0
    4       0
           ..
    5222    0
    5223    1
    5224    0
    5225    0
    5226    1
    Name: Label, Length: 5227, dtype: int64


```python
x=train
```


```python
x
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Dependents</th>
      <th>MonthlyCharges</th>
      <th>Partner</th>
      <th>PhoneService</th>
      <th>SeniorCitizen</th>
      <th>TotalCharges</th>
      <th>tenure</th>
      <th>Label</th>
      <th>Contract_One year</th>
      <th>...</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
      <th>gender_Male</th>
      <th>DeviceProtection_No</th>
      <th>DeviceProtection_Yes</th>
      <th>MultipleLines_No</th>
      <th>MultipleLines_Yes</th>
      <th>TVProgram_No</th>
      <th>TVProgram_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>24.150000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1505.900000</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>76.142284</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>946.581518</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>26.200000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1077.500000</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>24.650000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1138.800000</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>19.150000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>477.600000</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5222</th>
      <td>5222</td>
      <td>0</td>
      <td>20.750000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>485.200000</td>
      <td>23</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5223</th>
      <td>5223</td>
      <td>0</td>
      <td>19.900000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>19.900000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5224</th>
      <td>5224</td>
      <td>1</td>
      <td>111.750000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7511.300000</td>
      <td>68</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5225</th>
      <td>5225</td>
      <td>1</td>
      <td>78.350000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3211.200000</td>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5226</th>
      <td>5226</td>
      <td>1</td>
      <td>105.950000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5341.800000</td>
      <td>50</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5227 rows × 23 columns</p>
</div>


```python
x.shape
```


    (5227, 23)


```python
对训练集进行预处理 去掉label列
```


```python
x=x.drop(['Label'],axis=1)
```


```python
x
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Dependents</th>
      <th>MonthlyCharges</th>
      <th>Partner</th>
      <th>PhoneService</th>
      <th>SeniorCitizen</th>
      <th>TotalCharges</th>
      <th>tenure</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>...</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
      <th>gender_Male</th>
      <th>DeviceProtection_No</th>
      <th>DeviceProtection_Yes</th>
      <th>MultipleLines_No</th>
      <th>MultipleLines_Yes</th>
      <th>TVProgram_No</th>
      <th>TVProgram_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>24.150000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1505.900000</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>76.142284</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>946.581518</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>26.200000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1077.500000</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>24.650000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1138.800000</td>
      <td>45</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>19.150000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>477.600000</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5222</th>
      <td>5222</td>
      <td>0</td>
      <td>20.750000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>485.200000</td>
      <td>23</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5223</th>
      <td>5223</td>
      <td>0</td>
      <td>19.900000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>19.900000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5224</th>
      <td>5224</td>
      <td>1</td>
      <td>111.750000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7511.300000</td>
      <td>68</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5225</th>
      <td>5225</td>
      <td>1</td>
      <td>78.350000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3211.200000</td>
      <td>41</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5226</th>
      <td>5226</td>
      <td>1</td>
      <td>105.950000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5341.800000</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5227 rows × 22 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
```


```python
X_train
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Dependents</th>
      <th>MonthlyCharges</th>
      <th>Partner</th>
      <th>PhoneService</th>
      <th>SeniorCitizen</th>
      <th>TotalCharges</th>
      <th>tenure</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>...</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
      <th>gender_Male</th>
      <th>DeviceProtection_No</th>
      <th>DeviceProtection_Yes</th>
      <th>MultipleLines_No</th>
      <th>MultipleLines_Yes</th>
      <th>TVProgram_No</th>
      <th>TVProgram_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1547</th>
      <td>1547</td>
      <td>0</td>
      <td>109.950000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7634.250000</td>
      <td>69</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4458</th>
      <td>4458</td>
      <td>1</td>
      <td>90.450000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5957.900000</td>
      <td>65</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3637</th>
      <td>3637</td>
      <td>0</td>
      <td>100.350000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1358.850000</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4261</th>
      <td>4261</td>
      <td>0</td>
      <td>93.202343</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4317.382891</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5140</th>
      <td>5140</td>
      <td>0</td>
      <td>20.700000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20.700000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4931</th>
      <td>4931</td>
      <td>0</td>
      <td>96.329426</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>279.015779</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3264</th>
      <td>3264</td>
      <td>0</td>
      <td>94.973354</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>296.161668</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1653</th>
      <td>1653</td>
      <td>0</td>
      <td>101.392138</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4570.315257</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2607</th>
      <td>2607</td>
      <td>0</td>
      <td>46.200000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2431.950000</td>
      <td>54</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2732</th>
      <td>2732</td>
      <td>0</td>
      <td>93.450000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4872.200000</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>4181 rows × 22 columns</p>
</div>


```python
X_test
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Dependents</th>
      <th>MonthlyCharges</th>
      <th>Partner</th>
      <th>PhoneService</th>
      <th>SeniorCitizen</th>
      <th>TotalCharges</th>
      <th>tenure</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>...</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
      <th>gender_Male</th>
      <th>DeviceProtection_No</th>
      <th>DeviceProtection_Yes</th>
      <th>MultipleLines_No</th>
      <th>MultipleLines_Yes</th>
      <th>TVProgram_No</th>
      <th>TVProgram_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3359</th>
      <td>3359</td>
      <td>0</td>
      <td>109.210765</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7884.234551</td>
      <td>71</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>451</th>
      <td>451</td>
      <td>0</td>
      <td>49.200000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>216.900000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>502</th>
      <td>502</td>
      <td>1</td>
      <td>46.350000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1662.050000</td>
      <td>35</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>898</th>
      <td>898</td>
      <td>1</td>
      <td>79.400000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4820.550000</td>
      <td>61</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2760</th>
      <td>2760</td>
      <td>0</td>
      <td>85.300000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>424.150000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>1936</td>
      <td>0</td>
      <td>90.077391</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2879.093257</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>45</td>
      <td>0</td>
      <td>43.250000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>219.000000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2626</th>
      <td>2626</td>
      <td>1</td>
      <td>74.500000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4674.550000</td>
      <td>63</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>87</th>
      <td>87</td>
      <td>0</td>
      <td>89.946825</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>89.946825</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3895</th>
      <td>3895</td>
      <td>1</td>
      <td>60.000000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>60.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1046 rows × 22 columns</p>
</div>


```python
y_train
```


    1547    0
    4458    0
    3637    1
    4261    1
    5140    0
           ..
    4931    1
    3264    1
    1653    0
    2607    0
    2732    0
    Name: Label, Length: 4181, dtype: int64


```python
y_test
```


    3359    1
    451     1
    502     0
    898     0
    2760    1
           ..
    1936    1
    45      1
    2626    0
    87      1
    3895    1
    Name: Label, Length: 1046, dtype: int64



```python
test = pd.read_csv('C:/Users/15192/Desktop/pro_test(1).csv')
```


```python
test
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Dependents</th>
      <th>MonthlyCharges</th>
      <th>Partner</th>
      <th>PhoneService</th>
      <th>SeniorCitizen</th>
      <th>TotalCharges</th>
      <th>tenure</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>...</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
      <th>gender_Male</th>
      <th>DeviceProtection_No</th>
      <th>DeviceProtection_Yes</th>
      <th>MultipleLines_No</th>
      <th>MultipleLines_Yes</th>
      <th>TVProgram_No</th>
      <th>TVProgram_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5227</td>
      <td>1</td>
      <td>112.250000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>8041.650000</td>
      <td>72</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5228</td>
      <td>0</td>
      <td>45.100000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>45.100000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5229</td>
      <td>0</td>
      <td>59.100000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>772.850000</td>
      <td>14</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5230</td>
      <td>0</td>
      <td>19.650000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>19.650000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5231</td>
      <td>0</td>
      <td>19.250000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>855.100000</td>
      <td>48</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>6529</td>
      <td>0</td>
      <td>66.735121</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>300.711288</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>6530</td>
      <td>0</td>
      <td>40.808150</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>40.808150</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>6531</td>
      <td>0</td>
      <td>51.019882</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>51.019882</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>6532</td>
      <td>0</td>
      <td>76.046925</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>76.046925</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>6533</td>
      <td>0</td>
      <td>73.081768</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>969.402171</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1307 rows × 22 columns</p>
</div>


```python
y
```


    0       0
    1       1
    2       0
    3       0
    4       0
           ..
    5222    0
    5223    1
    5224    0
    5225    0
    5226    1
    Name: Label, Length: 5227, dtype: int64



### 使用决策树方法


```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
```


```python
X_train
```



</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Dependents</th>
      <th>MonthlyCharges</th>
      <th>Partner</th>
      <th>PhoneService</th>
      <th>SeniorCitizen</th>
      <th>TotalCharges</th>
      <th>tenure</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>...</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
      <th>gender_Male</th>
      <th>DeviceProtection_No</th>
      <th>DeviceProtection_Yes</th>
      <th>MultipleLines_No</th>
      <th>MultipleLines_Yes</th>
      <th>TVProgram_No</th>
      <th>TVProgram_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1547</th>
      <td>1547</td>
      <td>0</td>
      <td>109.950000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7634.250000</td>
      <td>69</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4458</th>
      <td>4458</td>
      <td>1</td>
      <td>90.450000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5957.900000</td>
      <td>65</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3637</th>
      <td>3637</td>
      <td>0</td>
      <td>100.350000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1358.850000</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4261</th>
      <td>4261</td>
      <td>0</td>
      <td>93.202343</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4317.382891</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5140</th>
      <td>5140</td>
      <td>0</td>
      <td>20.700000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20.700000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4931</th>
      <td>4931</td>
      <td>0</td>
      <td>96.329426</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>279.015779</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3264</th>
      <td>3264</td>
      <td>0</td>
      <td>94.973354</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>296.161668</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1653</th>
      <td>1653</td>
      <td>0</td>
      <td>101.392138</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4570.315257</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2607</th>
      <td>2607</td>
      <td>0</td>
      <td>46.200000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2431.950000</td>
      <td>54</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2732</th>
      <td>2732</td>
      <td>0</td>
      <td>93.450000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4872.200000</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>4181 rows × 22 columns</p>
</div>


```python
y_train
```




    1547    0
    4458    0
    3637    1
    4261    1
    5140    0
           ..
    4931    1
    3264    1
    1653    0
    2607    0
    2732    0
    Name: Label, Length: 4181, dtype: int64

**使用训练集测试精确度**


```python
pred_tree = clf.predict(X_test)
```


```python
print(accuracy_score(y_test,pred_tree))
```

    0.6912045889101338



```python
y_tree=clf.predict(test)
```


```python
y_tree
```

**对结果进行输出**


    array([0, 1, 0, ..., 1, 1, 1], dtype=int64)




```python
sub1 = pd.read_csv('C:/Users/15192/Desktop/sub_test.csv')
sub1
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5227</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5228</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5229</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5230</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5231</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>6529</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>6530</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>6531</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>6532</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>6533</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1307 rows × 2 columns</p>
</div>

**将结果保存在CSV中**


```python
sub1['Label']= y_tree
sub1 = sub1.set_index('ID')
sub1.to_csv('C:/Users/15192/Desktop/tree_result.csv')
```


```python
comptree = pd.read_csv('C:/Users/15192/Desktop/数据挖掘/大作业/tree_result.csv')
```


```python
collist = ['Label']
def binary_map(x):
    return x.map({1:'Yes',0:"No"})
comptree[collist] = comptree[collist].apply(binary_map)
```


```python
comptree
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5227</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5228</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5229</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5230</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5231</td>
      <td>No</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>6529</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>6530</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>6531</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>6532</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>6533</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>1307 rows × 2 columns</p>
</div>




```python
comptree.to_csv('C:/Users/15192/Desktop/数据挖掘/大作业/comptree.csv')
```





### 使用gbm方法



因为朴素贝叶斯方法精确度低 不适用 因此改用gbm方法


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



```python
gbm = LGBMClassifier()
gbm.fit(X_train,y_train)
pred_gbm = gbm.predict(X_test)
print(accuracy_score(y_test,pred_gbm))//对比精确度
```

    0.7820267686424475



```python
test = pd.read_csv('C:/Users/15192/Desktop/pro_test(1).csv')
test    
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Dependents</th>
      <th>MonthlyCharges</th>
      <th>Partner</th>
      <th>PhoneService</th>
      <th>SeniorCitizen</th>
      <th>TotalCharges</th>
      <th>tenure</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>...</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
      <th>gender_Male</th>
      <th>DeviceProtection_No</th>
      <th>DeviceProtection_Yes</th>
      <th>MultipleLines_No</th>
      <th>MultipleLines_Yes</th>
      <th>TVProgram_No</th>
      <th>TVProgram_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5227</td>
      <td>1</td>
      <td>112.250000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>8041.650000</td>
      <td>72</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5228</td>
      <td>0</td>
      <td>45.100000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>45.100000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5229</td>
      <td>0</td>
      <td>59.100000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>772.850000</td>
      <td>14</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5230</td>
      <td>0</td>
      <td>19.650000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>19.650000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5231</td>
      <td>0</td>
      <td>19.250000</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>855.100000</td>
      <td>48</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>6529</td>
      <td>0</td>
      <td>66.735121</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>300.711288</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>6530</td>
      <td>0</td>
      <td>40.808150</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>40.808150</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>6531</td>
      <td>0</td>
      <td>51.019882</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>51.019882</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>6532</td>
      <td>0</td>
      <td>76.046925</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>76.046925</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>6533</td>
      <td>0</td>
      <td>73.081768</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>969.402171</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1307 rows × 22 columns</p>
</div>


```python
y_gbm=gbm.predict(test)
y_gbm
```

**输出结果**


    array([0, 0, 0, ..., 1, 1, 1], dtype=int64)




```python
sub2 = pd.read_csv('C:/Users/15192/Desktop/sub_test.csv')
sub2
```



</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5227</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5228</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5229</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5230</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5231</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>6529</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>6530</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>6531</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>6532</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>6533</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1307 rows × 2 columns</p>
</div>

**转换成csv**


```python
sub2['Label']= y_gbm
sub2 = sub2.set_index('ID')
sub2.to_csv('C:/Users/15192/Desktop/gbm_result.csv')
```


```python
compgbm = pd.read_csv('C:/Users/15192/Desktop/数据挖掘/大作业/gbm_result.csv')
compgbm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5227</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5228</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5229</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5230</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5231</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>6529</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>6530</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>6531</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>6532</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>6533</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1307 rows × 2 columns</p>
</div>




```python
compgbm.to_csv('C:/Users/15192/Desktop/数据挖掘/大作业/compgbm.csv')
```


```python
collist = ['Label']
def binary_map(x):
    return x.map({1:'Yes',0:"No"})
compgbm[collist] = compgbm[collist].apply(binary_map)
compgbm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5227</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5228</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5229</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5230</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5231</td>
      <td>No</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>6529</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>6530</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>6531</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>6532</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>6533</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>1307 rows × 2 columns</p>
</div>




```python
compgbm.to_csv('C:/Users/15192/Desktop/数据挖掘/大作业/compgbm.csv')
```

