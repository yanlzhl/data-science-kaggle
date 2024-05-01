import pandas as pd
import numpy as np

#
# # 使用pandas加载数据
# data = pd.read_csv(data_path)
# data_columns = data.columns
#
# proto_type_dict = {"tcp":abs(np.cos(0)),"icmp":abs(np.cos(1)),"udp":abs(np.cos(2))}
# service_type_dict = {'mqtt':abs(np.cos(1)), 'dns':abs(np.cos(1)),'ntp':abs(np.cos(2)),
#                      'dhcp':abs(np.cos(3)),'ssl':abs(np.cos(4)),'ssh':abs(np.cos(5)),
#                      '-':abs(np.cos(6)),'http':abs(np.cos(7)),'radius':abs(np.cos(8)),
#                      'irc':abs(np.cos(9))}
# normal_attack_type = {'MQTT_Publish': np.array(1).astype(np.float32), 'Thing_Speak': np.array(1).astype(np.float32), 'Wipro_bulb': np.array(1).astype(np.float32)}
#
#
# # data convert
# data_ny = data.to_numpy()
# proto_type_column = data_ny[:, 3]
# service_type_column = data_ny[:, 4]
# attack_type_column = data_ny[:, -1]
#
# converted_proto_type_column = np.array([proto_type_dict.get(proto_type, proto_type) for proto_type in proto_type_column])
# converted_service_type_column = np.array([service_type_dict.get(service_type, service_type) for service_type in service_type_column])
# converted_attack_type_column =  np.array([normal_attack_type.get(attack_type, np.array(0).astype(np.float32)) for attack_type in attack_type_column])
#
# # 将转换后的列替换回原矩阵中
# data_ny[:, 3] = converted_proto_type_column
# data_ny[:, 4] = converted_service_type_column
# data_ny[:, -1] = converted_attack_type_column
# np.random.shuffle(data_ny)
#
# data = pd.DataFrame(data_ny.astype(np.float32), columns=data_columns)
#
#
# # 查看数据的前几行
# # print(data_ny[:, -1])
# # print(data.columns[5:84])
# # print(data.columns[1:84])
# # print(data.columns[84])
# # print(data.head())
#
# # # 检测有无空缺值
# # missing_values = ["?"]  # 定义缺失值的标识符
# # # 使用isnull()函数检测缺失值
# # missing_data = data.isin(missing_values)
# # # 统计每列缺失值的数量
# # missing_count = missing_data.sum()
# # print(missing_count)
#
# # 确定特征值
# x = data[data.columns[5:84]]
# y = data[data.columns[84]]
# # panda查看Y有多少种不同的值
# # print(data[data.columns[3]].unique())
# # print(data[data.columns[4]].unique())
# # print(data[data.columns[84]].unique())
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

print("Get data")
rt_iot2022 = fetch_ucirepo(id=942)
X = rt_iot2022.data.features
y = rt_iot2022.data.targets
print("Process data")
attack_labels = ['DOS_SYN_Hping', 'ARP_poisioning', 'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN', 'NMAP_OS_DETECTION',
                 'NMAP_TCP_scan', 'DDOS_Slowloris', 'Metasploit_Brute_Force_SSH', 'NMAP_FIN_SCAN']
y = y.isin(attack_labels).astype(int)

positive_samples = np.sum(y == 1)
negative_samples = np.sum(y == 0)
print(X)
print(y)
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = pd.concat([X.drop(non_numeric_columns, axis=1),
                       pd.DataFrame(encoder.fit_transform(X[non_numeric_columns]),
                                    columns=encoder.get_feature_names_out(input_features=non_numeric_columns))],
                      axis=1)
X_encoded.columns = X_encoded.columns.astype(str)

# 01 准备训练测试数据
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded,y, test_size=0.25,
                                                    random_state=2)

# 02 使用逻辑回归进行
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline



# normalization
ss =  MinMaxScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# no regularization  0.9773554256010396
lr_no = LogisticRegression(penalty=None)
# # regularization with L1
lr_l1 =  LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=100)
# # regularization with L2
lr_l2 =  LogisticRegression(penalty='l2', solver='liblinear')
# # L1-L2 regularization (elastic-net)
lr_elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.2)

# best amount of components is 10 after testing
pca = PCA(n_components=10)
log_reg = LogisticRegression(penalty=None)
lr_pca = Pipeline([('scaler', ss), ('pca', pca), ('log_reg', log_reg)])

lr_no.fit(X_train, y_train)
lr_l1.fit(X_train, y_train)
lr_l2.fit(X_train, y_train)
lr_elastic_net.fit(X_train, y_train)
lr_pca.fit(X_train, y_train)

lr_no_predict = lr_no.predict(X_test)
lr_l1_predict = lr_l1.predict(X_test)
lr_l2_predict = lr_l2.predict(X_test)
lr_elastic_net_predict = lr_elastic_net.predict(X_test)
lr_pca_predict = lr_pca.predict(X_test)

# 03 performance
from sklearn.metrics import classification_report
print('No regularization accuracy score：', lr_no.score(X_test, y_test))
print('L1 regularization (Lasso) accuracy score：', lr_l1.score(X_test, y_test))
print('L2 regularization (ridge) accuracy score：', lr_l2.score(X_test, y_test))
print('L1-L2 regularization (elastic-net) accuracy  score：', lr_elastic_net.score(X_test, y_test))
print('PCA dimensionality reduction accuracy score：', lr_pca.score(X_test, y_test))


print(classification_report(y_test, lr_no_predict, target_names=['0', '1']))
print(classification_report(y_test, lr_l1_predict, target_names=['0', '1']))
print(classification_report(y_test, lr_l2_predict, target_names=['0', '1']))
print(classification_report(y_test, lr_elastic_net_predict, target_names=['0', '1']))
print(classification_report(y_test, lr_pca_predict, target_names=['0', '1']))
