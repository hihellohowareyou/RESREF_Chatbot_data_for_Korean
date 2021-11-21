from transformers import BartForConditionalGeneration
from retrieval import DPR
import pandas as pd
from kobart import get_kobart_tokenizer

def generator():
    model = BartForConditionalGeneration.from_pretrained('generator/results')
    tokenizer = get_kobart_tokenizer()
    while True:
        sentence = '<s>' + input().strip() + '</s>'
        word = tokenizer(sentence, return_tensors='pt')
        x = model.generate(word['input_ids'], max_length=128, top_p=0.92, num_beams=5,
                           eos_token_id=tokenizer.eos_token_id,
                           bad_words_ids=[[tokenizer.unk_token_id, tokenizer.pad_token_id]], num_return_sequences=1,
                           no_repeat_ngram_size=2)[0]
        print(sentence)
        print(tokenizer.batch_decode(x.tolist(), skip_special_tokens=True))

def retriever():
    dpr = DPR()
    data = pd.read_csv('../Chatbot_data/ChatbotData.csv')
    dpr.models("klue/roberta-large", "../dpr/retrieval_models/encoder.pt")
    dpr.get_dense_embedding()
    dpr.build_faiss()
    print("안녕하세요")
    while True:
        sentence = '<s>' + input().strip() + '</s>'
        retrieve = dpr.get_relevant_doc(sentence)
        sentence = data['A'][retrieve[1][0]]
        print(sentence)

def RESREF():
    model = BartForConditionalGeneration.from_pretrained('retriever_and_refind/result')
    model.eval()
    # tokenizer = get_kobart_tokenizer()
    tokenizer = get_kobart_tokenizer()
    dpr = DPR()
    data = pd.read_csv('../Chatbot_data/ChatbotData.csv')
    dpr.models("klue/roberta-large", "../dpr/retrieval_models/encoder.pt")
    dpr.get_dense_embedding()
    dpr.build_faiss()
    print("안녕하세요")
    while True:
        sentence = '<s>' + input().strip() + '</s>'
        retrieve = dpr.get_relevant_doc(sentence)
        print(retrieve)
        sentence = sentence + '<s>' + data['A'][retrieve[1][0]] + '</s>'

        word = tokenizer(sentence, return_tensors='pt')
        x = model.generate(word['input_ids'], max_length=36, num_beams=5, min_length=5,
                           eos_token_id=tokenizer.eos_token_id,
                           bad_words_ids=[[tokenizer.unk_token_id, tokenizer.pad_token_id]], num_return_sequences=5,
                           no_repeat_ngram_size=2)
        print(sentence)
        print(tokenizer.batch_decode(x.tolist(), skip_special_tokens=True))
        print(tokenizer.batch_decode(x.tolist()[0][len(sentence):], skip_special_tokens=True))


def main(args):
    if args.chactbot_type == 'generator':
        generator()
    elif args.chactbot_type == 'retriever':
        retriever()
    else:
        RESREF()
    model = BartForConditionalGeneration.from_pretrained('retriever_and_refind/result')
    model.eval()
    # tokenizer = get_kobart_tokenizer()
    tokenizer = get_kobart_tokenizer()
    dpr = DPR(tokenize_fn=1)
    data = pd.read_csv('../Chatbot_data/ChatbotData.csv')
    dpr.models("klue/roberta-large","../dpr/retrieval_models/encoder.pt")
    dpr.get_dense_embedding()
    dpr.build_faiss()
    print("안녕하세요")
    while True:
        sentence = '<s>' + input().strip() + '</s>'
        retrieve = dpr.get_relevant_doc(sentence)
        print(retrieve)
        sentence = sentence + '<s>' + data['A'][retrieve[1][0]] + '</s>'

        word = tokenizer(sentence,return_tensors='pt')
        x = model.generate(word['input_ids'],max_length=36,num_beams=5,min_length=5,eos_token_id=tokenizer.eos_token_id,
                                                    bad_words_ids=[[tokenizer.unk_token_id,tokenizer.pad_token_id]],num_return_sequences=5,no_repeat_ngram_size=2)
        print(sentence)
        print(tokenizer.batch_decode(x.tolist(),skip_special_tokens=True))
        print(tokenizer.batch_decode(x.tolist()[0][len(sentence):],skip_special_tokens=True))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--chactbot_type", type=int, default=20, help="chatbot type you can get generator,retriever, RESREF")
