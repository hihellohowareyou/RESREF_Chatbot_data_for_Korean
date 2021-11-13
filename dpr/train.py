from transformers import AutoTokenizer,AdamW
import torch
from torch.utils.data import random_split,DataLoader,Dataset
from tqdm import tqdm
from model import BertEncoder
import pandas as pd
import torch.nn.functional as F

class dprDataset(Dataset):
    def __init__(self,path,tokenizer,max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        csv_file = pd.read_csv(path)
        print(f'before {len(csv_file)}')
        csv_file = csv_file.drop_duplicates(['A'])
        print(f'after {len(csv_file)}')
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

def train(args):
    q_encoder = BertEncoder.from_pretrained(args.model_name)
    p_encoder = BertEncoder.from_pretrained(args.model_name)
    q_encoder.to(args.device)
    p_encoder.to(args.device)
    no_decay = ["bias" ,"LayerNorm.weight"]
    dataset = dprDataset('../Chatbot_data/ChatbotData.csv',args.model_name)
    x = int(len(dataset) * 0.8)
    train_set, val_set = random_split(dataset, [x, len(dataset) - x])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    optimizer_grouped_parameters = [
        {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        # eps=args.adam_epsilon
    )
    for epoch in range(args.epochs):
        step = 0
        losses = 0
        for data in tqdm(train_loader):
            step += 1
            p_inputs = {'input_ids': data['q_input_ids'].to(args.device),
                        'attention_mask': data['q_attention_mask'].to(args.device)
                        }
            q_inputs = {'input_ids': data['a_input_ids'].to(args.device),
                        'attention_mask': data['a_attention_mask'].to(args.device)
                        }
            targets = torch.arange(0, len(p_inputs['input_ids'])).long().to(args.device)
            q_output = q_encoder(**q_inputs)
            p_output = p_encoder(**p_inputs)
            retrieval = torch.matmul(q_output, p_output.T)
            loss = F.cross_entropy(retrieval, targets)
            # retrieval_scores = F.log_softmax(retrieval, dim=1)
            #
            # loss = F.nll_loss(retrieval_scores, targets)
            losses += loss.item()
            q_encoder.zero_grad()
            p_encoder.zero_grad()

            loss.backward()
            optimizer.step()
            # scheduler.step()
            if step % 50 == 0:
                print(f'{epoch}epoch loss: {losses / (step)}')
        losses = 0
        correct = 0
        step = 0
        for data in tqdm(valid_loader):
            with torch.no_grad():
                p_inputs = {'input_ids': data['q_input_ids'].to(args.device),
                            'attention_mask': data['q_attention_mask'].to(args.device)
                            }
                q_inputs = {'input_ids': data['a_input_ids'].to(args.device),
                            'attention_mask': data['a_attention_mask'].to(args.device)
                            }

                targets = torch.arange(0, len(p_inputs['input_ids'])).long().to(args.device)
                q_output = q_encoder(**q_inputs)
                p_output = p_encoder(**p_inputs)

                retrieval = torch.matmul(q_output, p_output.T)
                retrieval_scores = F.log_softmax(retrieval, dim=1)
                retrieval_pred = torch.argmax(retrieval_scores, -1)
                # correct += (retrieval_pred == targets).float().sum().to('cpu')
                correct += torch.sum(retrieval_pred == targets).to("cpu")

                loss = F.nll_loss(retrieval_scores, targets)
                losses += loss.item()
                step += 1
        print(f'valid {epoch}epoch loss: {losses / (step)},acc:{correct/len(val_set)}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name',type=str,default='klue/bert-base',
                        help='model name')
    parser.add_argument('--dataset_name',type=str,default='../data/train_dataset', help='dataset')
    parser.add_argument('--warmup_steps',type=int,default=500, help='dataset')
    parser.add_argument('--weight_decay',type=float,default=0.01, help='dataset')
    parser.add_argument('--learning_rate',type=float,default=10e-5, help='dataset')
    parser.add_argument('--epochs',type=int,default=50, help='epochs')
    parser.add_argument('--device',type=str,default='cuda', help='device')
    parser.add_argument('--batch_size',type=str,default=20, help='batch_size')
    parser.add_argument('--seed',type=int,default=42, help='seed')

    args = parser.parse_args()
    train(args)
