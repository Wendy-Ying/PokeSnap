import pandas as pd

# 读取CSV文件
df = pd.read_csv("objects_single.csv")

# 打乱数据行
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# 拆分数据
split_index = int(0.1 * len(df_shuffled))
df_part1 = df_shuffled.iloc[:split_index]
df_part2 = df_shuffled.iloc[split_index:]

# 将拆分后的数据保存回CSV文件
df_part1.to_csv("objects_single_part1.csv", index=False)
df_part2.to_csv("objects_single_part2.csv", index=False)

print("Data have been shuffled and splitted into two files")
