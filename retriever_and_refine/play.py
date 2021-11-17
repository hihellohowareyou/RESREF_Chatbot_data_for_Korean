from transformers import AutoTokenizer,GPT2LMHeadModel,Seq2SeqTrainingArguments,DataCollatorWithPadding,GPT2LMHeadModel,Seq2SeqTrainer
from retrieval import DPR

model = GPT2LMHeadModel.from_pretrained('results')
tokenizer = tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
# tokenizer = get_kobart_tokenizer()
dpr = DPR(tokenize_fn='klue/bert-base')
dpr.models("klue/bert-base", "../dpr/retrieval_models/p_encoder.pt", "../dpr/retrieval_models/q_encoder.pt")
dpr.get_dense_embedding()
dpr.build_faiss()

sentence = '<s>친구가 서울에서 보자는데 서울은 너무 멀어.</s>'
retrieve = dpr.get_relevant_doc_faiss(sentence)
sentence = '<s>' + retrieve + '/s' + sentence

word = tokenizer(sentence,return_tensors='pt')
x = model.generate(word['input_ids'],max_length=128,num_beams=10,eos_token_id=tokenizer.eos_token_id,
                                            bad_words_ids=[[tokenizer.unk_token_id,tokenizer.pad_token_id]],num_return_sequences=5,no_repeat_ngram_size=2)
print(sentence)
print(tokenizer.batch_decode(x.tolist(),skip_special_tokens=True))