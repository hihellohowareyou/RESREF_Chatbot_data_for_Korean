##구조
- train.py
retriever and refine으로 학습하는 곳, <answer><quesion><answer>의 형태로 들어옴
- dataset.py
dataset을 만드는 곳,<answer><quesion><answer>의 형태로 만듦, 단 일정 비율은 실제 정답이 아니라
retrievl한 데이터를 앞의 answer부분에 넣음
- play.py 실제 실험해보는 곳, context가 주어지면 retriever로 가장 가까운 답을 찾아 <retrive><context>의 구조로 model에 넣음
