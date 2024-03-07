import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 'Onion':[1,0,0,1,1,1] 是6次购物，1代表购买了onion，0表示没有
data = {'ID':[1,2,3,4,5,6],
       'Onion':[1,0,0,1,1,1],
       'Potato':[1,1,0,1,1,1],
       'Burger':[1,1,0,0,1,1],
       'Milk':[0,1,1,1,0,1],
       'Beer':[0,0,1,0,1,0]}

df =  pd.DataFrame(data)
print(df.shape)

df = df[['ID', 'Onion', 'Potato', 'Burger', 'Milk', 'Beer' ]]
print(df)

# calculate the support of each food
frequent_itemsets = apriori(df[['Onion', 'Potato', 'Burger', 'Milk', 'Beer' ]], min_support=0.50, use_colnames=True)
print(frequent_itemsets)

# 可以指定不同的衡量标准与最小阈值
rule  = association_rules(frequent_itemsets,metric='lift',min_threshold=1)
print(rule)
