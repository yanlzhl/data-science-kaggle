import pandas as pd

# 创建一个示例数据集
data = {'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e']}
df = pd.DataFrame(data)

# 随机打乱数据集
df_shuffled = df.sample(frac=1, random_state=42)

print("原始数据集：")
print(df)

print("\n随机打乱后的数据集：")
print(df_shuffled)
