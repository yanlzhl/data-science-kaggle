# from ucimlrepo import fetch_ucirepo
# rt_iot2022 = fetch_ucirepo(id=942)
# X = rt_iot2022.data.features
# y = rt_iot2022.data.targets
# # print(type(X))
# # print(type(y))
# print(rt_iot2022.metadata)
# print(rt_iot2022.variables)


import pandas as pd
import numpy as np
import requests
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 从网页读取文件
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
response = requests.get(url)
data = response.text
df = pd.read_csv(StringIO(data), header=None)
df
"""
	0	1	2	3	4	5	6	7	8	9	10
0	1000025	5	1	1	1	2	1	3	1	1	2
1	1002945	5	4	4	5	7	10	3	2	1	2
2	1015425	3	1	1	1	2	2	3	1	1	2
3	1016277	6	8	8	1	3	4	3	7	1	2
4	1017023	4	1	1	3	2	1	3	1	1	2
...	...	...	...	...	...	...	...	...	...	...	...
694	776715	3	1	1	1	3	2	1	1	1	2
695	841769	2	1	1	1	2	1	1	1	1	2
696	888820	5	10	10	3	7	3	8	10	2	4
697	897471	4	8	6	4	3	4	10	6	1	4
698	897471	4	8	8	5	4	5	10	4	1	4
699 rows × 11 columns
"""

# 检测有无空缺值
missing_values = ["?"]  # 定义缺失值的标识符

# 使用isnull()函数检测缺失值
missing_data = df.isin(missing_values)

# 统计每列缺失值的数量
missing_count = missing_data.sum()
print(missing_count)
"""
0      0
1      0
2      0
3      0
4      0
5      0
6     16
7      0
8      0
9      0
10     0
dtype: int64
"""

# 缺失值处理
# 替换缺失值
df = df.replace(to_replace='?', value=np.NAN)
# 丢弃缺失值的数据
df = df.dropna()

# 确定特征值
x = df.iloc[:, 1:9]
y = df.iloc[:, 10]
# 标准化
standar = StandardScaler()
x = standar.fit_transform(x)
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2, test_size=0.2)
# 机器学习：逻辑回归
estimator = LogisticRegression(penalty="l2")  # 默认就是l2惩罚
estimator.fit(x_train, y_train)

# 模型评估
# 准确率
scores = estimator.score(x_test, y_test)
print(f"准确率为：{scores * 100}%")

# 预测值
y_predict = estimator.predict(x_test)
print(f"预测值为：\r\n{y_predict}")
"""
准确率为：94.16058394160584%
预测值为：
[4 4 2 4 2 2 2 2 2 4 2 2 4 2 2 2 4 4 2 2 2 2 4 4 4 2 2 4 2 4 4 4 4 2 4 2 4
 2 2 4 2 2 4 2 2 4 2 2 4 2 2 2 4 2 2 2 2 2 4 4 2 2 2 2 2 2 4 4 2 4 2 2 2 2
 2 2 2 2 4 4 4 4 4 2 4 4 4 2 4 2 2 4 4 4 2 4 2 4 2 2 4 2 2 2 2 4 4 2 2 2 2
 2 4 4 2 2 4 2 2 2 2 2 2 4 2 2 4 4 4 4 2 4 2 4 2 2 2]
"""