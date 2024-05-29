import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import sys
sys.setrecursionlimit(3000)
#构建Bert的输入数据格式

# 读取包含打乱IPC分类号的CSV文件
# 假设IPC分类号存储在名为"IPC_Classification"的列中
# 定义一个字典来映射IPC分类号到对应的ID
# 分类：
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
ipc_to_id_mapping1 = {
    "A": list(range(0, 16)),
    "B": list(range(16, 54)),
    "C": list(range(54, 75)),
    "D": list(range(75, 84)),
    "E": list(range(84, 92)),
    "F": list(range(92, 110)),
    "G": list(range(110, 125)),
    "H": list(range(125, 132)),
}
ipc_to_id_mapping_updated = {
    "A01": 0, "A21": 1, "A22": 2, "A23": 3, "A24": 4, "A41": 5, "A42": 6, "A43": 7, "A44": 8, "A45": 9,
    "A46": 10, "A47": 11, "A61": 12, "A62": 13, "A63": 14,
    "B01": 15, "B02": 16, "B03": 17, "B04": 18, "B05": 19, "B06": 20, "B07": 21, "B08": 22, "B09": 23,
    "B21": 24, "B22": 25, "B23": 26, "B24": 27, "B25": 28, "B26": 29, "B27": 30, "B28": 31, "B29": 32,
    "B30": 33, "B31": 34, "B32": 35, "B33": 36, "B41": 37, "B42": 38, "B43": 39, "B44": 40, "B60": 41,
    "B61": 42, "B62": 43, "B63": 44, "B64": 45, "B65": 46, "B66": 47, "B67": 48, "B68": 49, "B81": 50,
    "B82": 51,
    "C01": 52, "C02": 53, "C03": 54, "C04": 55, "C05": 56, "C06": 57, "C07": 58, "C08": 59, "C09": 60,
    "C10": 61, "C11": 62, "C12": 63, "C13": 64, "C14": 65, "C21": 66, "C22": 67, "C23": 68, "C25": 69,
    "C30": 70, "C40": 71,
    "D01": 72, "D02": 73, "D03": 74, "D04": 75, "D05": 76, "D06": 77, "D07": 78, "D21": 79,
    "E01": 80, "E02": 81, "E03": 82, "E04": 83, "E05": 84, "E06": 85, "E21": 86,
    "F01": 87, "F02": 88, "F03": 89, "F04": 90, "F15": 91, "F16": 92, "F17": 93, "F21": 94, "F22": 95,
    "F23": 96, "F24": 97, "F25": 98, "F26": 99, "F27": 100, "F28": 101, "F41": 102, "F42": 103,
    "G01": 104, "G02": 105, "G03": 106, "G04": 107, "G05": 108, "G06": 109, "G07": 110, "G08": 111,
    "G09": 112, "G10": 113, "G11": 114, "G12": 115, "G16": 116, "G21": 117,
    "H01": 118, "H02": 119, "H03": 120, "H04": 121, "H05": 122, "H10": 123,
}



# 定义一个函数来根据IPC分类号映射到对应的ID
def map_ipc_to_id(ipc_number):
    if isinstance(ipc_number, str):
        prefix = ipc_number[:3]  # 获取IPC分类号的前三个字符作为前缀
        return ipc_to_id_mapping_updated.get(prefix, -1)  # 使用字典.get()方法进行映射，如果没有找到则返回-1或者其他你认为合适的值
    else:
        return -1  # 如果ipc_number不是字符串类型，返回-1或者其他你认为合适的值

def read_data(data_dir):
    data = pd.read_csv(data_dir)
    data['专利名称'] = data['专利名称']
    data['摘要'] = data['摘要']
    # 将IPC分类号映射为对应的ID并存储到新的列中
    data['分类标签'] = data['IPC分类号'].apply(map_ipc_to_id)
    data['text'] = data['专利名称'] + data['摘要']
    return data
def fill_paddings(data, maxlen):
    '''补全句长，缺的补0'''
    if len(data) < maxlen:
        pad_len = maxlen-len(data)
        paddings = [0 for _ in range(pad_len)]
        data = torch.tensor(data + paddings)
    else:
        data = torch.tensor(data[:maxlen])
    return data

class InputDataSet(Dataset):
    ''''''
    def __init__(self,data,tokenizer,max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def classes(self):
        return self.labels

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, item):  # item是索引 用来取数据
        text = str(self.data['text'][item])
        labels = self.data['分类标签'][item]
        position_ids = self.data['专利名称'][item]
        labels = torch.tensor(labels, dtype=torch.long)#int-->long
        # 手动构建
        tokens = self.tokenizer.tokenize(text)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = [101] + tokens_ids + [102]
        input_ids = fill_paddings(tokens_ids,self.max_len)

        attention_mask = [1 for _ in range(len(tokens_ids))]
        attention_mask = fill_paddings(attention_mask,self.max_len)

        token_type_ids = [0 for _ in range(len(tokens_ids))]
        token_type_ids = fill_paddings(token_type_ids,self.max_len)

        return {
            'text': text,#文本数据
            'input_ids': input_ids,#vocab数字标记
            'attention_mask': attention_mask,#全1标记
            'token_type_ids': token_type_ids,#全0标记
            'labels': labels, #要输出的标签
            'position_ids': position_ids#要输出的专利号
        }