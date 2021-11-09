from torch.utils.data import Dataset
import pandas as pd
from kobart import get_kobart_tokenizer

def add_to(sentenct,bos_token='<s>',eos_token='</s>'):
    new_sentence = bos_token + sentenct + eos_token
    return new_sentence
class wellnessDataset(Dataset):
    def __init__(self,path,max_seq_len=256):
        self.max_seq_len = max_seq_len
        self.tokenizer = get_kobart_tokenizer()
        self.bos_token = self.tokenizer.bos_token
        print(self.bos_token)
        self.eos_token = self.tokenizer.eos_token
        csv_file = pd.read_csv(path)
        self.quesion = csv_file['Q'].map(add_to)
        self.answer = csv_file['A'].map(add_to)
        print(self.quesion)

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, idx):
        question = self.tokenizer(self.quesion[idx])
        answer = self.tokenizer(self.answer[idx])
        return {'input_ids':question['input_ids'],'attention_mask':question['attention_mask'],
                'decoder_input_ids':answer['input_ids'],'decoder_attention_mask':answer['attention_mask']}

if __name__ == '__main__':
    x = wellnessDataset('../Chatbot_data/ChatbotData.csv')
    print(len(x))
    p = x[1]
    tokenizer = get_kobart_tokenizer()
    print(tokenizer.decode(p['input_ids']))
    print(tokenizer.decode(p['decoder_input_ids'], skip_special_tokens=True))
