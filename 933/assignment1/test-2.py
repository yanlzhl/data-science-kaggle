# -*- coding: utf-8 -*-

import pickle 
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# fetch dataset
rt_iot2022 = fetch_ucirepo(id=942)

# data (as pandas dataframes)
X = rt_iot2022.data.features
y = rt_iot2022.data.targets
metadata = rt_iot2022.metadata
variables = rt_iot2022.variables

# The next lines of code are only informative/diagnostic
print(rt_iot2022.metadata) # this allows you to see the dataset metadata
print(rt_iot2022.variables) # allows you to see the variables



Dic =  {'data_x':rt_iot2022.data.features, 'data_y': rt_iot2022.data.targets, 'variables':rt_iot2022.variables}
with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(Dic, f)

with open('saved_dictionary.pkl', 'rb') as f:
    Dic = pickle.load(f)
    
    
X = Dic['data_x']
y = Dic['data_y']
variables = Dic['variables']

del Dic

# process labels  abnormal_ratio：87.12%
labels = set([item[0] for item in y.values.tolist()])
normalLabels = ['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb']
targets = [0  if item[0] in normalLabels else 1 for item in y.values.tolist()]
num_normal = len([x for x in targets if x == 0])
num_abnormal = len(targets) - num_normal
abnormal_ratio = num_abnormal / len(targets)


proto_type = set(X['proto'].tolist())
serive_type = set(X['service'].tolist())
proto_type_dict = {"tcp":abs(np.cos(0)),"icmp":abs(np.cos(1)),"udp":abs(np.cos(2))}
service_type_dict = {'mqtt':abs(np.cos(1)), 'dns':abs(np.cos(1)),'ntp':abs(np.cos(2)),
                     'dhcp':abs(np.cos(3)),'ssl':abs(np.cos(4)),'ssh':abs(np.cos(5)),
                     '-':abs(np.cos(6)),'http':abs(np.cos(7)),'radius':abs(np.cos(8)),
                     'irc':abs(np.cos(9))}
sources = X.to_dict()

def convert_targets(t):
    out = []
    for i, item in enumerate(t):
        out.append(item)
    out = np.array(out).astype(np.float32)
    return out[:,None]
        
def generate_dataset(sources):
    assert type(sources) == dict
    output = sources
    for key, value in output['proto'].items():
        output['proto'][key] = proto_type_dict[value]
    for key, value in output['service'].items():
        output['service'][key] = service_type_dict[value]
        
    del output['id.orig_p']
    del output['id.resp_p']
    
    out = []
    for key, dic in output.items():
        row = []
        for k, v in dic.items():
            row.append(v)  
        row = np.array(row)
        row = normalize(row[None, :], norm='max', axis=1)[0]
        out.append(row)
    out = np.array(out)
    return out

output = np.transpose(generate_dataset(sources))
target = convert_targets(targets)
del sources, targets
dataset = np.concatenate((output, target), axis=1)
np.random.shuffle(dataset)
num_train = int(0.8 * dataset.shape[0])
trainset = dataset[:num_train,:]
valset = dataset[num_train:,:]



# PCA dimension reduction
pca = PCA(n_components=10)
pca.fit(output)
data = pca.transform(output)
dataset_reduced = np.concatenate((data, target), axis=1)
np.random.shuffle(dataset_reduced)
trainset_reduced = dataset_reduced[:num_train,:]
valset_reduced = dataset_reduced[num_train:,:]





# penalty: None, l1, l2, elasticnet              solver: newton-cg, lbfgs, liblinear, sag, saga
# The ‘newton-cg’, ‘sag’, and ‘lbfgs’ solvers support only L2 regularization with primal formulation,
# or no regularization. The ‘liblinear’ solver supports both L1 and L2 regularization,
# with a dual formulation only for the L2 penalty. The Elastic-Net regularization is only supported by the ‘saga’ solver.

clf1 = LogisticRegression(penalty=None).fit(trainset[:,:-1], trainset[:,-1])
clf2 = LogisticRegression(penalty='l1', solver='liblinear').fit(trainset[:,:-1], trainset[:,-1])
clf3 = LogisticRegression(penalty='l2', solver='liblinear').fit(trainset[:,:-1], trainset[:,-1])
clf4 = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.2).fit(trainset[:,:-1], trainset[:,-1])
clf5 = LogisticRegression(penalty=None).fit(trainset_reduced[:,:-1], trainset_reduced[:,-1])




yp1 = clf1.predict(valset[:,:-1])
score1 = accuracy_score(valset[:,-1], yp1)


yp2 = clf2.predict(valset[:,:-1])
score2 = accuracy_score(valset[:,-1], yp2)


yp3 = clf3.predict(valset[:,:-1])

score3 = accuracy_score(valset[:,-1], yp3)


yp4 = clf4.predict(valset[:,:-1])
score4 = accuracy_score(valset[:,-1], yp4)


yp5 = clf5.predict(valset_reduced[:,:-1])
score5 = accuracy_score(valset_reduced[:,-1], yp5)

print(score1, score2, score3, score4, score5)
























