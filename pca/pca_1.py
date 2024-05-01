import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA


# 用你下载的数据文件的路径替换这里的路径
data_path = '../data/RT_IOT2022.csv'

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

# # 使用pandas加载数据
# # data = pd.read_csv(data_path)
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


X = data[data.columns[5:84]]
y = data[data.columns[84]]

# https://zhuanlan.zhihu.com/p/257144872
# plt.style.use("ggplot")
# X, y = make_classification(
#     n_samples=1000,  # 1000个观测值
#     n_features=50,  # 50个特征
#     n_informative=10,  # 只有10个特征有预测能力
#     n_redundant=40,  # 40个冗余的特征
#     n_classes=2,  # 目标变量包含两个类别
#     random_state=123  # 随机数种子，保证可重复结果
# )

# 创建PCA对象，声明主成分个数
pca = PCA(n_components=10)
# 拟合数据
res = pca.fit_transform(X)

print("original shape: ", X.shape)
print("transformed shape: ", res.shape)
print(res[:10,:3])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(res[:, 0], res[:, 1])
ax.set_title("RT_IOT2022 - Relationship between principal components", fontsize=15)
ax.set_xlabel("First Main Component")
ax.set_ylabel("Second Main Component")
plt.show()


# 当主成分个数等于5，仅仅保留了原始变异的78.8%，这显然是不够的，那么如何正确地选择主成分的个数？
var_ratio = pca.explained_variance_ratio_
for idx, val in enumerate(var_ratio, 1):
    print("Principle component %d: %.2f%%" % (idx, val * 100))
print("total: %.2f%%" % np.sum(var_ratio * 100))


# n_components=None, 主成分个数默认等于特征数量
pca = PCA(n_components=None)
pca.fit(X)
evr = pca.explained_variance_ratio_ * 100 # 获取解释方差比率
fig, ax = plt.subplots(figsize=(10, 7)) # 查看累计解释方差比率与主成分个数的关系
ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), "-ro")
ax.set_title("RT_IOT2022-Cumulative Explained Variance Ratio", fontsize=15)
ax.set_xlabel("number of components")
ax.set_ylabel("explained variance ratio(%)")
plt.show()


# 另一种更简单的办法是，设定累计解释方差比率的目标，让sklearn自动选择最优的主成分个数。
target = 0.99  # 保留原始数据集90%的变异
pca = PCA(n_components=target)
X_pca = pca.fit_transform(X)
# res = PCA(n_components=target).fit_transform(X)
print(" original shape: ", X.shape)
print(" transformed shape: ", res.shape)

# 获取主成分的载荷（特征权重）
loadings = pca.components_
# 输出主成分的载荷，每一行对应一个主成分，每一列对应一个原始特征
print("Principal Components Loadings:")
print(loadings)
# 假设原始特征有列名，可以通过以下代码获取列名
# 假设列名保存在一个名为 feature_names 的列表中
# 请根据你的实际情况调整这一部分代码
feature_names = ['id.orig_p', 'id.resp_p', 'proto', 'service',
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

# 输出每个主成分对应的原始特征名称
print("\nOriginal Features Contributing to Each Principal Component:")
for i, component in enumerate(loadings):
    print(f"Principal Component {i+1}:")
    for j, weight in enumerate(component):
           if weight > 0.005:
              print(f"{feature_names[j]}: {weight:.4f}")
    print()
