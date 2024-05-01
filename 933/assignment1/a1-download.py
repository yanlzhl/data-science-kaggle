import pandas as pd
import numpy as np

# https://archive.ics.uci.edu/dataset/942/rt-iot2022

# 1. elasticnet中的比例问题 需要通过交叉验证来获得最优值
# 2. PCA中将维 选定某一个列 来操作，需要测试？

# 用你下载的数据文件的路径替换这里的路径
data_path = '../../data/RT_IOT2022.csv'

# 根据官方数据构建类别
column_names = ['id.orig_p', 'id.resp_p', 'proto', 'service',
       'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_data_pkts_tot',
       'bwd_data_pkts_tot', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec',
       'flow_pkts_per_sec', 'down_up_ratio', 'fwd_header_size_tot',
       'fwd_header_size_min', 'fwd_header_size_max', 'bwd_header_size_tot',
       'bwd_header_size_min', 'bwd_header_size_max', 'flow_FIN_flag_count',
       'flow_SYN_flag_count', 'flow_RST_flag_count', 'fwd_PSH_flag_count',
       'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'fwd_URG_flag_count',
       'bwd_URG_flag_count', 'flow_CWR_flag_count', 'flow_ECE_flag_count',
       'fwd_pkts_payload.min', 'fwd_pkts_payload.max', 'fwd_pkts_payload.tot',
       'fwd_pkts_payload.avg', 'fwd_pkts_payload.std', 'bwd_pkts_payload.min',
       'bwd_pkts_payload.max', 'bwd_pkts_payload.tot', 'bwd_pkts_payload.avg',
       'bwd_pkts_payload.std', 'flow_pkts_payload.min',
       'flow_pkts_payload.max', 'flow_pkts_payload.tot',
       'flow_pkts_payload.avg', 'flow_pkts_payload.std', 'fwd_iat.min',
       'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std',
       'bwd_iat.min', 'bwd_iat.max', 'bwd_iat.tot', 'bwd_iat.avg',
       'bwd_iat.std', 'flow_iat.min', 'flow_iat.max', 'flow_iat.tot',
       'flow_iat.avg', 'flow_iat.std', 'payload_bytes_per_second',
       'fwd_subflow_pkts', 'bwd_subflow_pkts', 'fwd_subflow_bytes',
       'bwd_subflow_bytes', 'fwd_bulk_bytes', 'bwd_bulk_bytes',
       'fwd_bulk_packets', 'bwd_bulk_packets', 'fwd_bulk_rate',
       'bwd_bulk_rate', 'active.min', 'active.max', 'active.tot', 'active.avg',
       'active.std', 'idle.min', 'idle.max', 'idle.tot', 'idle.avg',
       'idle.std', 'fwd_init_window_size', 'bwd_init_window_size',
       'fwd_last_window_size','Attack_type']

# 使用pandas加载数据
data = pd.read_csv(data_path)
data_columns = data.columns

proto_type_dict = {"tcp":abs(np.cos(0)),"icmp":abs(np.cos(1)),"udp":abs(np.cos(2))}
service_type_dict = {'mqtt':abs(np.cos(1)), 'dns':abs(np.cos(1)),'ntp':abs(np.cos(2)),
                     'dhcp':abs(np.cos(3)),'ssl':abs(np.cos(4)),'ssh':abs(np.cos(5)),
                     '-':abs(np.cos(6)),'http':abs(np.cos(7)),'radius':abs(np.cos(8)),
                     'irc':abs(np.cos(9))}
normal_attack_type = {'MQTT_Publish': np.array(1).astype(np.float32), 'Thing_Speak': np.array(1).astype(np.float32), 'Wipro_bulb': np.array(1).astype(np.float32)}


# data convert
data_ny = data.to_numpy()
proto_type_column = data_ny[:, 3]
service_type_column = data_ny[:, 4]
attack_type_column = data_ny[:, -1]

converted_proto_type_column = np.array([proto_type_dict.get(proto_type, proto_type) for proto_type in proto_type_column])
converted_service_type_column = np.array([service_type_dict.get(service_type, service_type) for service_type in service_type_column])
converted_attack_type_column =  np.array([normal_attack_type.get(attack_type, np.array(0).astype(np.float32)) for attack_type in attack_type_column])

data_ny[:, 3] = converted_proto_type_column
data_ny[:, 4] = converted_service_type_column
data_ny[:, -1] = converted_attack_type_column
np.random.shuffle(data_ny)

# convert data type to float32
data = pd.DataFrame(data_ny.astype(np.float32), columns=data_columns)


# 查看数据的前几行
# print(data_ny[:, -1])
# print(data.columns[5:84])
# print(data.columns[1:84])
# print(data.columns[84])
# print(data.head())

# # 检测有无空缺值
# missing_values = ["?"]  # 定义缺失值的标识符
# # 使用isnull()函数检测缺失值
# missing_data = data.isin(missing_values)
# # 统计每列缺失值的数量
# missing_count = missing_data.sum()
# print(missing_count)

# 确定特征值
x = data[data.columns[1:84]]
y = data[data.columns[84]]
# panda查看Y有多少种不同的值
# print(data[data.columns[3]].unique())
# print(data[data.columns[4]].unique())
# print(data[data.columns[84]].unique())



# 01 准备训练测试数据
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25,
                                                    random_state=2)


# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# no regularization  0.9773554256010396
lr = LogisticRegression(penalty=None)
# regularization with L1
# lr =  LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=100)
# regularization with L2
# lr =  LogisticRegression(penalty='l2', solver='liblinear')
# 跳用LogisticRegression中的fit函数／模块来训练模型参数
# L1-L2 regularization (elastic-net)
# lr = LogisticRegressionCV(penalty='elasticnet', solver='saga', l1_ratios=[0.2])

# pca = PCA(n_components=5)  # 保留95%的方差
# log_reg = LogisticRegression()
# lr = Pipeline([('scaler', StandardScaler()), ('pca', pca), ('log_reg', log_reg)])

lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

# 03 性能分析
from sklearn.metrics import classification_report
# 利用逻辑斯蒂回归自带的评分函数score获得模型在测试集上的准确定结果
print('精确率为：', lr.score(X_test, y_test))

# print(classification_report(y_test, lr_y_predict, target_names=['0', '1']))
# print(classification_report(y_test, lr_y_predict, target_names=['MQTT_Publish','Thing_Speak','Wipro_bulb','ARP_poisioning',
#  'DDOS_Slowloris','DOS_SYN_Hping','Metasploit_Brute_Force_SSH',
#  'NMAP_FIN_SCAN','NMAP_OS_DETECTION','NMAP_TCP_scan','NMAP_UDP_SCAN',
#  'NMAP_XMAS_TREE_SCAN']))
