import pandas as pd
import numpy as np

# 根据官方数据构建类别
column_names = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
data = pd.read_csv("../../data/breast-cancer-wisconsin.csv")

# 将？替换成标准缺失值表示
# data = data.replace(to_replace='?', value=np.nan)

# 丢弃带有缺失值的数据（只要一个维度有缺失）
# data = data.dropna(how='any')
print(data.shape)
# print(column_names[0:31])
# print(data.columns)



# 01 准备训练测试数据
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score

# print(column_names[1:10])
# print(data[column_names[1:10]])
# print( data[column_names[10]])

X_train, X_test, y_train, y_test = train_test_split(data[column_names[0:31]], data[column_names[0]], test_size=0.25,
                                                    random_state=42)
# # 查看训练和测试样本的数量和类别分布
print(y_train.value_counts())
print(y_test.value_counts())

# X_train：训练集的特征数据，包含了用于训练模型的样本数据。
# X_test：测试集的特征数据，包含了用于评估模型性能的样本数据，但这些数据在训练模型时是未曾见过的。
# y_train：训练集的目标变量数据，对应于训练集的特征数据，用于训练模型。
# y_test：测试集的目标变量数据，对应于测试集的特征数据，用于评估模型性能。



# 02 使用逻辑回归进行良／恶性肿瘤预测任务
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# 初始化 LogisticRegression
# lr = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
# lr = LogisticRegression(solver='liblinear', penalty='l1')
lr =  LogisticRegression(solver='lbfgs', penalty='l2')
# 跳用LogisticRegression中的fit函数／模块来训练模型参数
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)


# 03 性能分析
from sklearn.metrics import classification_report
# 利用逻辑斯蒂回归自带的评分函数score获得模型在测试集上的准确定结果
print('精确率为：', lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Maligant']))
