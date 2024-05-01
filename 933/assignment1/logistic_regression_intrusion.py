import pandas as pd
import numpy as np
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

# convert data
positive_samples = np.sum(y == 1)
negative_samples = np.sum(y == 0)
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = pd.concat([X.drop(non_numeric_columns, axis=1),
                       pd.DataFrame(encoder.fit_transform(X[non_numeric_columns]),
                                    columns=encoder.get_feature_names_out(input_features=non_numeric_columns))],
                      axis=1)
X_encoded.columns = X_encoded.columns.astype(str)

# 01 Training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded,y, test_size=0.25,
                                                    random_state=2)

# 02 predict
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# normalization
ss =  MinMaxScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# no regularization 0.9876868096166341
lr_no = LogisticRegression(penalty=None)
# regularization with L1  0.9871994801819364
lr_l1 =  LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=100)
# regularization with L2  0.9816764132553606
lr_l2 =  LogisticRegression(penalty='l2', solver='liblinear')
# L1-L2 regularization (elastic-net) 0.9823586744639377
lr_elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.2)

pca = PCA(n_components=10)  # 0.9667641325536063
log_reg = LogisticRegression()
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

