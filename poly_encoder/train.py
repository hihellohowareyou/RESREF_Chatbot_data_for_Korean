"""
dataset 불러오기

"""
from transformers import AutoTokenizer
from poly_encoder import polyEncoder,polyDataset
import torch
from torch.utils.data import random_split,DataLoader
from tqdm import tqdm
model = polyEncoder()
model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
dataset = polyDataset('../Chatbot_data/ChatbotData.csv')
x = int(len(dataset) * 0.8)
train_set, val_set = random_split(dataset, [x,len(dataset) - x])
train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
valid_loader = DataLoader(val_set,batch_size=32,shuffle=True)
optim = torch.optim.Adam(params=model.parameters(),lr=2e-5)
losses = 0

epochs = 10
for epoch in range(epochs):
    losses = 0
    corrects = 0
    for idx,iters in enumerate(tqdm(train_loader)):
        iter = {i:v.to('cuda') for i,v in iters.items()}
        loss,correct = model(context_input_ids=iter['q_input_ids'],responses_input_ids=iter['a_input_ids'],
              context_attention_masks=iter['q_attention_mask'],responses_attention_masks=iter['a_attention_mask'],labels=1)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses += loss.item()
        corrects += correct
    print(f'train{epoch}epochs loss:{losses / (idx + 1)},correct:{corrects / (64*(idx + 1))}')
    losses = 0
    corrects = 0
    for idx, iters in enumerate(tqdm(valid_loader)):
        iter = {i:v.to('cuda') for i,v in iters.items()}

        with torch.no_grad():
            loss,correct = model(context_input_ids=iter['q_input_ids'], responses_input_ids=iter['a_input_ids'],
                         context_attention_masks=iter['q_attention_mask'],
                         responses_attention_masks=iter['a_attention_mask'],labels=1)
            target = torch.arange(0,len(iter['q_input_ids']),dtype=torch.long).to('cuda')
            losses += loss.item()
            corrects += correct
    print(f'eval{epoch}epochs loss:{losses /(idx + 1)},correct:{corrects / (64*(idx + 1))}')
