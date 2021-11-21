git # jaebot


RESREF(retriever and refine) 구조의 챗봇입니다.
generator와 refine의 백본 모델은 skta1dml kobart모델을 사용했습니다.
retriever모델은 klue/roberta-large 모델을 사용했습니다.
모든 모델은 chatbot_data를 이용해 학습했습니다.

retriever, generator, generate and refine 구조의 챗봇을 체험해 볼 수 있습니다.
단 retriever 모델은 dpr/train.py로 retriever모델을 학습한 이후 사용할 수 있고 generator모델은 generator/train.py에서 generator 모델을 학습한 후에 사용할 수 있습니다.
RESREF모델은 dpr/train.py로 retriever모델을 학습하고 train.py로 refine 모델을 학습한 이후에 사용할 수 있습니다.
