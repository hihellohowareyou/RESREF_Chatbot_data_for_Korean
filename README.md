
RESREF(retriever and refine) 구조의 챗봇입니다.
단 원 논문과 다르게 retriever로 dpr을 사용했고 generator로 bart를 사용했습니다.
generator와 refine의 백본 모델은 skta1dml kobart모델을 사용했습니다.(https://github.com/SKT-AI/KoBART)
retriever모델은 klue/roberta-large 모델을 사용했습니다. (https://huggingface.co/klue/roberta-large)
모든 모델은 chatbot_data를 이용해 학습했습니다.(https://github.com/songys/Chatbot_data)

retriever, generator, generate and refine 구조의 챗봇을 체험해 볼 수 있습니다.
단 retriever 모델은 dpr/train.py로 retriever모델을 학습한 이후 사용할 수 있고 generator모델은 generator/train.py에서 generator 모델을 학습한 후에 사용할 수 있습니다.
RESREF모델은 dpr/train.py로 retriever모델을 학습하고 train.py로 refine 모델을 학습한 이후에 사용할 수 있습니다.


## reference
Retrieve and Refine: Improved Sequence Generation Models For Dialogue
https://arxiv.org/abs/1808.04776
Recipes for building an open-domain chatbot
https://arxiv.org/abs/2004.13637
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
https://arxiv.org/abs/1910.13461
skt ai -kobart
https://github.com/SKT-AI/KoBART
Chatbot_data_for_Korean v1.0
https://github.com/songys/Chatbot_data
