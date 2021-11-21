from torch.utils.data import Dataset
import pandas as pd
from kobart import get_kobart_tokenizer
def add_to_special_tokens(sentenct, bos_token='<s>', eos_token='</s>'):
    """ add to special token in """
    new_sentence = bos_token + sentenct + eos_token
    return new_sentence

class generatorDataset(Dataset):
    def __init__(self,path,max_seq_len=256):
        self.max_seq_len = max_seq_len
        self.tokenizer = get_kobart_tokenizer()
        self.bos_token = self.tokenizer.bos_token
        print(self.bos_token)
        self.eos_token = self.tokenizer.eos_token
        csv_file = pd.read_csv(path)
        self.quesion = csv_file['Q'].map(add_to_special_tokens)
        self.quesion = self.tokenizer(self.quesion.tolist(),padding=True,return_tensors='pt')

        self.answer = csv_file['A'].map(add_to_special_tokens)
        self.answer = self.tokenizer(self.answer.tolist(),padding=True,return_tensors='pt')
        print(self.quesion)

    def __len__(self):
        return len(self.answer['input_ids'])

    def __getitem__(self, idx):

        question = {i:v[idx] for i,v in self.quesion.items()}
        answer = {i:v[idx] for i,v in self.answer.items()}
        labels = answer['input_ids'][:]
        return {'input_ids':question['input_ids'],'attention_mask':question['attention_mask'],
                'labels':labels}

