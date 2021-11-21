import random
from torch.utils.data import Dataset
import pandas as pd
from kobart import get_kobart_tokenizer
from transformers import AutoTokenizer
from tqdm.tqdm import tqdm
from retrieval import DPR

def add_to_special_tokens(sentenct, bos_token='<s>', eos_token='</s>'):
    """ add to special token in """
    new_sentence = bos_token + sentenct + eos_token
    return new_sentence

class retrieveAndRefineDateset(Dataset):
    def __init__(self,data_path):
        # x.models("klue/bert-base", "../dpr/retrieval_models/p_encoder.pt", "../dpr/retrieval_models/q_encoder.pt")
        # x.get_dense_embedding()
        self.tokenizer = get_kobart_tokenizer()
        # x.build_faiss()
        data = pd.read_csv(data_path)
        self.concat = data['A'].map(add_to_special_tokens) + data['Q'].map(add_to_special_tokens)
        self.question = data['Q']
        self.answer = data['A']
        self.random_retirver()
        self.concat = self.concat.map(add_to_special_tokens)

        print(self.concat)
        self.concat = self.tokenizer(self.concat.tolist(),padding=True,return_tensors='pt')
        self.answer = self.tokenizer(self.answer.tolist(),padding=True,return_tensors='pt')

    def random_retirver(self):
        x = DPR(tokenize_fn='klue/roberta-large')
        x.models("klue/roberta-large", "../dpr/retrieval_models/encoder.pt")
        x.get_dense_embedding()
        x.build_faiss()
        for i in tqdm(range(len(self.question))):
            if random.random() <0.3:
                _,retrieve = x.get_relevant_doc(self.question[i])
                retrieve = self.answer[retrieve[0]]
                self.concat[i] = self.question[i] +'</s><s>' + retrieve
        x.del_models()

    def __len__(self):
        return len(self.concat['input_ids'])

    def __getitem__(self, idx):

        question = {i:v[idx] for i,v in self.concat.items()}
        answer = {i:v[idx] for i,v in self.answer.items()}
        labels = answer['input_ids']
        return {'input_ids':question['input_ids'],'attention_mask':question['attention_mask'],
                # 'decoder_input_ids':answer['input_ids'],'decoder_attention_mask':answer['attention_mask'],
                'labels':labels}
