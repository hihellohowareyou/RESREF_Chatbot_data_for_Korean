
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertModel, ElectraTokenizer,ElectraTokenizer,AutoTokenizer

class bert:
    def __init__(self):
        pass

class polyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")
        hidden_size = 768
        self.poly_m = 3
        self.poly_code_embeddings = nn.Embedding(self.poly_m, hidden_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight,hidden_size ** -0.5)

    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output

    def forward(self, context_input_ids, responses_input_ids,context_attention_masks=None,
                            responses_attention_masks=None, labels=None):
        # during training, only select the first response
        # we are using other instances in a batch as negative examples
        if labels is not None:
            responses_input_ids = responses_input_ids.unsqueeze(1)
            if responses_attention_masks is not None:
                responses_attention_masks = responses_attention_masks.unsqueeze(1)
        batch_size, res_cnt, seq_length = responses_input_ids.shape # res_cnt is 1 during training

        # context encoder
        ctx_out = self.bert(context_input_ids, context_attention_masks)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]

        # response encoder
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        if responses_attention_masks is not None:
            responses_attention_masks = responses_attention_masks.view(-1, seq_length)
        cand_emb = self.bert(responses_input_ids, responses_attention_masks)[0][:,0,:] # [bs, dim]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]

        # merge
        if labels is not None:
            # we are recycling responses for faster training
            # we repeat responses for batch_size times to simulate test phase
            # so that every context is paired with batch_size responses
            cand_emb = cand_emb.permute(1, 0, 2) # [1, bs, dim]
            cand_emb = cand_emb.expand(batch_size, batch_size, cand_emb.shape[2]) # [bs, bs, dim]
            ctx_emb = self.dot_attention(cand_emb, embs, embs).squeeze() # [bs, bs, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1) # [bs, bs]
            target = torch.arange(batch_size, dtype=torch.long).to(context_input_ids.device)
            # mask = torch.eye(batch_size).to(context_input_ids.device) # [bs, bs]
            loss = F.cross_entropy(dot_product,target)
            retrieval_pred = torch.argmax(dot_product, -1)

            correct = torch.sum(retrieval_pred == target).to("cpu")
            return loss,correct
        else:
            ctx_emb = self.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1)
            return dot_product

from torch.utils.data import Dataset
import pandas as pd
from kobart import get_kobart_tokenizer
"""
정답이 겹치는 경우가 있어서 이를 처리해주어야한다."""

class polyDataset(Dataset):
    def __init__(self,path,max_seq_len=256):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        csv_file = pd.read_csv(path)
        print(f'before {len(csv_file)}')
        csv_file = csv_file.drop_duplicates(['A'])
        print(f'before {len(csv_file)}')
        self.quesion = csv_file['Q']
        self.quesion = self.tokenizer(self.quesion.tolist(),padding=True,return_tensors='pt')
        self.answer = csv_file['A']

        self.answer = self.tokenizer(self.answer.tolist(),padding=True,return_tensors='pt')

    def __len__(self):
        return len(self.answer['input_ids'])

    def __getitem__(self, idx):

        question = {i:v[idx] for i,v in self.quesion.items()}
        answer = {i:v[idx] for i,v in self.answer.items()}
        return {'q_input_ids':question['input_ids'],'q_attention_mask':question['attention_mask'],
                'a_input_ids':answer['input_ids'],'a_attention_mask':answer['attention_mask']
                }

if __name__ == '__main__':
    x = polyDataset('../Chatbot_data/ChatbotData.csv')
    print(len(x))
    p = x[1]
    tokenizer = get_kobart_tokenizer()
    print(p['Q'])
    print(tokenizer.decode(p['Q']['input_ids']))
    print(tokenizer.decode(p['A']['input_ids']))
