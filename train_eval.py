import torch
import torch.nn as nn
import numpy as np
import time
import os
import logging
# 由于在pytorch里,所以用py标准代码torch.optim库,有常用优化算法如SGD和Adam
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers.utils.notebook import format_time

from modeling import BertForSeq
from data_load import InputDataSet, read_data, fill_paddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')
# device = torch.device('cpu')

def train(batch_size,EPOCHS):#一次输入样本数目+训练轮数
        # 模型输入model和bert_model
    # 导入包，加载预训练模型
    model_name='bert-base-chinese'
    model = BertForSeq.from_pretrained(model_name)
    # 导入配置文件
    model_config = BertConfig.from_pretrained(model_name)
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    # 通过配置和路径导入模型
    bert_model = BertModel.from_pretrained(model_name, config=model_config)
    bert_model = BertForSeq.from_pretrained(model_name, config=model_config)

# 模型数据利用tokernizer做初步的embedding处理
     # 通过词典导入分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    #输入数据
    train = read_data('data/data.csv')
    val = read_data('data/val_data.csv')
    # 训练集和验证集通过data_process类进行embedding封装，添加batch维度
    train_dataset = InputDataSet(train, tokenizer, 512)
    val_dataset = InputDataSet(val, tokenizer, 512)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        # 定义AdamW优化器
    optimizer = AdamW(model.parameters(), lr=2e-6, eps=1e-8)
    total_steps = len(train_dataloader) * EPOCHS  # 训练步数 len(dataset)*epochs / batchsize
    # 调度器，控制训练,用学习率预热API时，num_warmup_steps这个参数一定要设置为0，一定要设置为0，一定要设置为0！！！否则模型不会收敛
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

        # 时间变量，打印时间
    total_t0 = time.time()
    # log_creater控制台打印数据，再写进log文件
    log = log_creater(output_dir='./cache/logs/')
    log.info("   Train batch size = {}".format(batch_size))
    log.info("   Total steps = {}".format(total_steps))
    log.info("   Training Start!")
# 训练部分，轮次循环
    for epoch in range(EPOCHS):

        # 定义两个变量，用于存储训练集的准确率和损失
        total_train_loss = 0
        t0 = time.time()
        model.to(device)# 如果要进gpu运行，模型一定和数据位置一致
        model.train()
        for step, batch in enumerate(train_dataloader): #每个批次分配一个索引 step。然后可以在循环中使用 step 来跟踪当前批次的索引
            # 前向传播、计算损失、反向传播、更新参数等操作
            optimizer.zero_grad() # 清除掉之前batch的gradients
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)# 通过模型得到输出
            loss = outputs.loss  # 计算损失
            # 更新模型和调度器参数
            loss.backward()
            # 在backward得到梯度之后，step ()更新之前，使用梯度剪裁。
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度剪裁，防止梯度爆炸，梯度大于1会累乘
            optimizer.step()

            total_train_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{EPOCHS}, Step {step + 1}/{len(train_dataloader)}, Training Loss: {total_train_loss / (step + 1)}')
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader) # 求平均损失
        print("avg_train_loss=",avg_train_loss)
        log.info('====Epoch:[{}/{}] avg_train_loss={:.5f}===='.format(epoch+1,EPOCHS,avg_train_loss))
        log.info('====Training epoch took: {:}===='.format(format_time(time.time() - t0)))
        log.info('Running Validation...')

        avg_val_loss, avg_val_acc = evaluate(model, val_dataloader)# 通过函数evaluate得到要验证数据的损失和准确率，这里验证数据=初始数据
        print("avg_val_loss=", avg_val_loss)
        print("avg_val_acc=", avg_val_acc)
        log.info('====Epoch:[{}/{}] avg_val_loss={:.5f} avg_val_acc={:.5f}===='.format(epoch+1,EPOCHS,avg_val_loss,avg_val_acc))
        log.info('====Validation epoch took: {:}===='.format(format_time(time.time() - t0)))
        log.info('')

        if epoch == EPOCHS-1:
            # torch.save(model,'./cache/model.bin')
            model.save_pretrained('./cache/model3/')
            print('Model Saved!')
    log.info('')
    log.info('   Training Completed!')
    print('Total training took{:} (h:mm:ss)'.format(format_time(time.time() - total_t0)))

# model.eval(): 设置模型为评估模式，这会关闭 dropout 和 batch normalization。
# total_val_loss: 用于存储总的验证损失。
# correct_predictions: 用于计算总的正确预测数量。
# 使用 torch.no_grad() 上下文管理器，确保在评估过程中不计算梯度。
# 对于数据加载器中的每个批次，将输入数据和标签移到 GPU（如果可用），并进行模型推理。
# 从模型输出中提取损失和预测 logits。
# 累积总的验证损失。
# 使用 torch.argmax() 获取预测的类别索引，并将预测 logits 转移到 CPU 并转换为 NumPy 数组，以便进行后续的比较。
# 将标签转移到 CPU 并转换为 NumPy 数组。
# 计算总的正确预测数量。
# 计算平均验证准确率和平均验证损失，并将其返回。
def evaluate(model, val_dataloader):
    model.eval()
    total_val_loss = 0
    corrects = []
    for step, batch in enumerate(val_dataloader): #每个批次分配一个索引 step。然后可以在循环中使用 step 来跟踪当前批次的索引
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        position_ids = batch['position_ids']
        # print(position_ids)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        logits = torch.argmax(outputs.logits, dim=1)
        preds = logits.detach().cpu().numpy()
        labels_ids = labels.to('cpu').numpy()
        corrects.append((preds == labels_ids).mean())
        loss = outputs.loss
        total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_acc = np.mean(corrects)
    return avg_val_loss, avg_val_acc
def log_creater(output_dir): # 日志输出部分代码
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)
    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log
if __name__ == '__main__':
    train(batch_size=2,EPOCHS=7)