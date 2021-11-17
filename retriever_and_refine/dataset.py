from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from retrieval import DPR
import random
from tqdm import tqdm

def add_to(sentence,bos_token='<s>',eos_token='</s>'):
    new_sentence = bos_token + sentence + eos_token
    return new_sentence

class rertDateset(Dataset):
    def __init__(self,data_path,tokenizer):
        x = DPR(tokenize_fn='klue/bert-base')
        x.models("klue/bert-base", "../dpr/retrieval_models/p_encoder.pt", "../dpr/retrieval_models/q_encoder.pt")
        x.get_dense_embedding()
        x.build_faiss()
        print(x.p_embedding)
        data = pd.read_csv(data_path)

        docs = x.retrieve_faiss(data)
        print(docs)

def concat(sentence):
    new_sentence = sentence + '</s><s>'
    return new_sentence

class concatDateset(Dataset):
    def __init__(self,data_path,tokenizer):
        x = DPR(tokenize_fn='klue/bert-base')
        # x.models("klue/bert-base", "../dpr/retrieval_models/p_encoder.pt", "../dpr/retrieval_models/q_encoder.pt")
        # x.get_dense_embedding()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # x.build_faiss()
        data = pd.read_csv(data_path)
        self.concat = data['A'].map(concat) + data['Q']
        self.question = data['Q']
        self.answer = data['A']
        self.random_retirver()
        self.answer = data['A'].map(add_to)
        self.concat = self.concat.map(add_to)
        print(self.concat)
        # docs = x.get_relevant_doc_bulk_faiss(data['A'].tolist())
        self.data = self.tokenizer(self.concat.tolist(),self.answer.tolist(),padding=True,return_tensors='pt')

    def random_retirver(self):
        x = DPR(tokenize_fn='klue/bert-base')
        x.models("klue/bert-base", "../dpr/retrieval_models/p_encoder.pt", "../dpr/retrieval_models/q_encoder.pt")
        x.get_dense_embedding()
        x.build_faiss()
        for i in tqdm(range(len(self.question))):
            if random.random() <0.4:
                _,retrieve = x.get_relevant_doc(self.question[i])
                retrieve = self.answer[retrieve[0]]
                retrieve = retrieve + '</s><s>'
                self.concat[i] = retrieve + self.question[i]
        x.del_models

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):

        question = {i:v[idx] for i,v in self.data.items()}
        return {'input_ids':question['input_ids'],'attention_mask':question['attention_mask'],
                'labels':question['input_ids']}

if __name__ == '__main__':
    x = concatDateset('../Chatbot_data/ChatbotData.csv','skt/ko-gpt-trinity-1.2B-v0.5')
    print(x[0])