import pandas as pd
import numpy as np

# 创建一个示例的 NumPy 数组
array = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 将 NumPy 数组转换为 Pandas 的 DataFrame
df = pd.DataFrame(array, columns=['A', 'B', 'C'])

print(df)
