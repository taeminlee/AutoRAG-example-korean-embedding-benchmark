import pandas as pd
from datasets import Dataset, DatasetDict
from itertools import chain


def conv_corpus(corpus_parquet_path='data/ocr_corpus_v3.parquet'):
    # Parquet 파일 읽어오기
    corpus_df = pd.read_parquet(corpus_parquet_path)

    # 필요한 칼럼 선택
    corpus_df = corpus_df[['doc_id', 'contents']]

    # 칼럼 이름 변경 및 새로운 'title' 칼럼 생성
    corpus_df.columns = ['_id', 'text']
    corpus_df['title'] = ''

    # DataFrame을 HuggingFace Dataset으로 변환
    corpus_dataset = Dataset.from_pandas(corpus_df)
    
    return DatasetDict({'corpus': corpus_dataset})


def conv_queries(qa_parquet_path='data/qa_v4.parquet'):
    # Parquet 파일 읽기
    queries_df = pd.read_parquet(qa_parquet_path)

    # 필요한 칼럼 선택
    queries_df = queries_df[['qid', 'query']]

    # 칼럼 이름 변경
    queries_df.columns = ['_id', 'text']

    # DataFrame을 HuggingFace Dataset으로 변환
    queries_dataset = Dataset.from_pandas(queries_df)

    return DatasetDict({'queries': queries_dataset})


def conv_qrels(qa_parquet_path='data/qa_v4.parquet'):
    # Parquet 파일 읽기
    queries_df = pd.read_parquet('data/qa_v4.parquet')

    # 행 당 열(qid)과 retrieval_gt 가져오기
    qid_list = queries_df['qid'].tolist()
    retrieval_gt_list = queries_df['retrieval_gt'].tolist()

    # 2중 배열을 평탄화
    flat_retrieval_gt_list = list(chain.from_iterable(retrieval_gt_list))

    # 평탄화된 각 원소에 대해 record 생성 후 다시 데이터프레임으로 묶기
    record_list = []
    for idx, elements in enumerate(flat_retrieval_gt_list):
        for element in elements:
            record = {'query-id': qid_list[idx], 'corpus-id': element, 'score': 1.0}
            record_list.append(record)

    # 새로운 DataFrame 만들기
    qrels_df = pd.DataFrame(record_list)

    # DataFrame을 HuggingFace Dataset으로 변환
    qrels_dataset = Dataset.from_pandas(qrels_df)

    return DatasetDict({'test': qrels_dataset})


if __name__ == '__main__':
    corpus_dataset = conv_corpus()
    queries_dataset = conv_queries()
    qrels_dataset = conv_qrels()

    from huggingface_hub import HfApi
    api = HfApi()

    repo_id = "nlpai-lab/markers_bm"

    api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)

    corpus_dataset.push_to_hub(repo_id, "corpus")
    queries_dataset.push_to_hub(repo_id, "queries")
    qrels_dataset.push_to_hub(repo_id, "default")
