
from transformers import BertTokenizer, BertModel, BertPreTrainedModel

from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

class BertForSeq(BertPreTrainedModel):
    def __init__(self,config):  ##  config.json
        super(BertForSeq,self).__init__(config)
        self.num_labels = 124 # 类别数目,标签数目A-H大类的124个项目
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids=None,  # 输入的id,模型会帮你把id转成embedding
                attention_mask=None, # attention里的mask
                token_type_ids=None, # [CLS]A[SEP]B[SEP] 就这个A还是B, 有的话就全1, 没有就0
                position_ids=None,  # 专利号位置id
                labels = None, # 做分类时需要的label
                return_dict = None
                ):
        # input_ids：输入文本的embedding后结果， attention_mask：注意力编码； token_type_ids：分句id, labels：标签0-4， return_dict：预测值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict #输出类型tensor OR string
        ## loss损失 预测值preds
        outputs = self.bert(input_ids, # 文本数据依照vocab的id
                            attention_mask=attention_mask, # 全1标记
                            token_type_ids=token_type_ids, # 全0标记
                            position_ids=position_ids, # 专利名称
                            return_dict=return_dict    # 预测值编号A-H
                            )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # softmax前的输入
        # 计算loss，初始化loss为None
        loss = None
        if labels is not None:
            # cross entropy 是用来衡量两个概率分布之间的距离的，softmax能把一切转换成概率分布，那么自然二者经常在一起使用。
            # CrossEntropyLoss()损失函数，是将Log_Softmax()激活函数与NLLLoss()损失函数的功能综合在一起了
            # 创建交叉熵损失函数对象 多类别分类
            criterion = nn.CrossEntropyLoss()
            # 计算损失logits:输入张量；labels：目标类别
            loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,  ##损失
            logits=logits,  #softmax层的输入
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )