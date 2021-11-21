import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from sklearn.feature_extraction.text import TfidfVectorizer
from  transformers import AutoTokenizer
from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

from model import BertEncoder

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class DPR:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../Chatbot_data/ChatbotData.csv",
    ) -> NoReturn:

        self.data_path = data_path
        csv_file = pd.read_csv(data_path)
        self.contexts = csv_file['A'].tolist()
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def del_models(self):
        del self.p_encoder
        del self.q_encoder

    def models(self,model_name,p_encoder_path,q_encoder_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            stride=64,
            padding=True
        )
        self.p_encoder = BertEncoder.from_pretrained(model_name)
        self.p_encoder.load_state_dict(torch.load(p_encoder_path))
        self.p_encoder.to('cuda')

        self.q_encoder = BertEncoder.from_pretrained(model_name)
        self.q_encoder.load_state_dict(torch.load(q_encoder_path))
        self.q_encoder.to('cuda')

    def get_dense_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"DPR.bin"
        emd_path = pickle_name
        # change here
        print("Build passage embedding")
        # print("Embedding pickle saved.")
        # print(self.p_embedding.shape)
        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
            print(type(self.p_embedding))
        else:
            print("Build passage embedding")
            p_seqs = self.tokenizer(self.contexts, padding=True, return_tensors='pt')
            wiki_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'])
            wiki_loader = DataLoader(wiki_dataset, batch_size=20)
            for idx, data in enumerate(tqdm(wiki_loader)):
                p_inputs = {'input_ids': data[0].to('cuda'),
                            'attention_mask': data[1].to('cuda'),
                            }
                with torch.no_grad():
                    p_inputs = {k: v for k, v in p_inputs.items()}
                    output = self.p_encoder(**p_inputs)
                    if idx == 0:
                        self.p_embedding = output
                    else:
                        self.p_embedding = torch.cat((self.p_embedding, output), 0)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = indexer_name
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.cpu().detach().numpy()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)
            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_INNER_PRODUCT
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")


    def get_relevant_doc(self, query: str, k: Optional[int] = 10) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        q_input = self.tokenizer(query, padding="max_length", truncation=True, return_tensors='pt')
        q_input = {i: v.to('cuda') for i, v in q_input.items()}
        query_vec = self.q_encoder(**q_input)
        assert (
            torch.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = torch.mm(query_vec, self.p_embedding.T)
        if not isinstance(result, np.ndarray):
            result = result.cpu().detach().numpy()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        q_seqs = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt')
        q_dataset = TensorDataset(q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])
        print(len(q_dataset))
        q_loader = DataLoader(q_dataset,batch_size=4)
        for idx,q_input in enumerate(tqdm(q_loader)):
            q_inputs = {'input_ids': q_input[0],
                        'attention_mask': q_input[1],
                        'token_type_ids': q_input[2]}
            q_input = {i:v.to('cuda') for i,v in q_inputs.items()}
            with torch.no_grad():
                output = self.q_encoder(**q_input)
                if idx == 0:
                    query_vec = output
                else:
                    query_vec = torch.cat((query_vec, output), 0)


        assert (
            torch.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = torch.mm(query_vec,self.p_embedding.T)
        if not isinstance(result, np.ndarray):
            result = result.cpu().detach().numpy()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        queries = query_or_dataset["Q"]
        total = []

        with timer("query faiss search"):
            doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                queries.tolist(), k=topk
            )
        for idx, example in tqdm(query_or_dataset.iterrows()):
            print(example['Q'])
            tmp = {
                # Query와 해당 id를 반환합니다.
                "Q": example["Q"],
                "A":example['A'],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "retrieve": " ".join(
                    [self.contexts[pid] for pid in doc_indices[idx]]
                ),
            }
            total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            q_input = self.tokenizer(query, padding="max_length", truncation=True, return_tensors='pt')
            q_input = {i: v.to('cuda') for i, v in q_input.items()}
            query_vec = self.q_encoder(**q_input)
        assert (
            torch.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        if not isinstance(query_vec, np.ndarray):
            query_vec = query_vec.cpu().detach().numpy()
        with timer("query faiss search"):
            D, I = self.indexer.search(query_vec, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        q_seqs = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt')
        q_dataset = TensorDataset(q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])
        q_loader = DataLoader(q_dataset,batch_size=4)
        for idx,q_input in enumerate(tqdm(q_loader)):
            q_inputs = {'input_ids': q_input[0],
                        'attention_mask': q_input[1]
                        }
            q_input = {i:v.to('cuda') for i,v in q_inputs.items()}
            with torch.no_grad():
                output = self.q_encoder(**q_input)
                if idx == 0:
                    query_vec = output
                else:
                    query_vec = torch.cat((query_vec, output), 0)

        assert (
            torch.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        if not isinstance(query_vec, np.ndarray):
            query_vec = query_vec.cpu().detach().numpy()
        D, I = self.indexer.search(query_vec, k)

        return D.tolist(), I.tolist()

if __name__ == '__main__':
    x = DPR(tokenize_fn='klue/bert-base')
    x.models("klue/bert-base","retrieval_models/p_encoder.pt","retrieval_models/q_encoder.pt")
    x.get_dense_embedding()
    x.build_faiss()
    doc = '.'
    answer = x.get_relevant_doc(doc,5)
    csv_file = pd.read_csv("../Chatbot_data/ChatbotData.csv")
    answers = csv_file['A']
    for i in answer[1]:
        print(answers[i])