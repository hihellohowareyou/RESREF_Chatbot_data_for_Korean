from transformers import AutoModel,modeling_outputs,RobertaPreTrainedModel, BertPreTrainedModel,BertModel, RobertaModel,PreTrainedModel,DistilBertPreTrainedModel,DistilBertModel,ElectraModel,ElectraPreTrainedModel
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        # self.model = BertModel.from_pretrained(config)
        self.bert = BertModel(config)
        self.layer = nn.Linear(768,2048)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask
                            )
        pooled_output = outputs[1]
        # pooled_output = self.layer(pooled_output)
        # result = torch.sum(pooled_output,1)/ 768
        return pooled_output
