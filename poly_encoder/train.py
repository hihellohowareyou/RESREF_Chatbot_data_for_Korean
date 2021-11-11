"""
dataset 불러오기

"""
from transformers import AutoTokenizer
from poly_encoder import polyEncoder,polyDataset
import torch
from torch.utils.data import random_split,DataLoader
from tqdm import tqdm
model = polyEncoder()
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
dataset = polyDataset('../Chatbot_data/ChatbotData.csv')
x = int(len(dataset) * 0.8)
train_set, val_set = random_split(dataset, [x,len(dataset) - x])
dataloader = DataLoader(train_set,batch_size=16,shuffle=True)
optim = torch.optim.Adam(params=model.parameters())
losses = 0
for idx,iter in enumerate(tqdm(dataloader)):

    loss = model(context_input_ids=iter['q_input_ids'],responses_input_ids=iter['a_input_ids'],
          context_attention_masks=iter['q_attention_mask'],responses_attention_masks=iter['a_attention_mask'],labels=1)
    optim.zero_grad()
    loss.backward()
    optim.step()
    losses += loss.item()
    if idx % 100 == 0 and idx != 0:
        print(losses/(idx+1))