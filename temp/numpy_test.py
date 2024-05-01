import numpy as np

# 创建一个示例矩阵
matrix = np.array([[1, 2, 'China'],
                    [4, 5, 'US'],
                    [7, 8, 'India'],
                    [10, 11, 'China'],
                    [13, 14, 'Japan']])  # 添加了一个不在转换范围内的示例

# 获取矩阵的第3列
third_column = matrix[:, 2]

# 创建一个转换字典
conversion_dict = {'China': 1, 'US': 2, 'India': 3}

# 使用字典映射将字符转换为对应的值
converted_column = np.array([conversion_dict.get(country, country) for country in third_column])

# 将转换后的列替换回原矩阵中
matrix[:, 2] = converted_column

np.random.shuffle(matrix)

print(matrix)
