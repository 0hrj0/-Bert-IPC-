import pandas as pd

ipc_to_id_mapping = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
}
def map_ipc_to_id(ipc_number):
    if isinstance(ipc_number, str):
        prefix = ipc_number[:1]  # 获取IPC分类号的前三个字符作为前缀
        return ipc_to_id_mapping.get(prefix, -1)  # 使用字典.get()方法进行映射，如果没有找到则返回-1或者其他你认为合适的值
    else:
        return -1  # 如果ipc_number不是字符串类型，返回-1或者其他你认为合适的值

def count_records(data_dir):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    count0 = 0
    data = pd.read_csv(data_dir)
    data['分类标签'] = data['IPC分类号'].apply(map_ipc_to_id)
    for index, row in data.iterrows():
        if row['分类标签'] == 0:
            count0 += 1
        elif row['分类标签'] == 1:
            count1 += 1
        elif row['分类标签'] == 2:
            count2 += 1
        elif row['分类标签'] == 3:
            count3 += 1
        elif row['分类标签'] == 4:
            count4 += 1
        elif row['分类标签'] == 5:
            count5 += 1
        elif row['分类标签'] == 6:
            count6 += 1
        elif row['分类标签'] == 7:
            count7 += 1
    print("类别 A 的条数:", count0)
    print("类别 B 的条数:", count1)
    print("类别 C 的条数:", count2)
    print("类别 D 的条数:", count3)
    print("类别 E 的条数:", count4)
    print("类别 F 的条数:", count5)
    print("类别 G 的条数:", count6)
    print("类别 H 的条数:", count7)

# 用法示例
csv_file = 'data.csv'  # 将文件路径替换为你的 CSV 文件路径
count_records(csv_file)






import matplotlib.pyplot as plt

# 类别及其对应的条目数
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
counts = [7960, 17929, 9387, 3540, 3669, 7559, 7566, 4589]

# 创建柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, counts, color='skyblue')

# 在每个柱子的顶部显示数目
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100, str(count), ha='center', va='bottom')

# 添加标题和标签
plt.title('条目数分布')
plt.xlabel('类别')
plt.ylabel('条数')

# 展示图形
plt.xticks(rotation=45)  # 旋转 x 轴标签
plt.tight_layout()  # 调整布局，防止标签重叠
plt.show()

