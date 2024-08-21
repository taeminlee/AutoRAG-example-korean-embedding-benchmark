import os

import autorag
import click
from autorag.evaluator import Evaluator
from dotenv import load_dotenv
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.upstage import UpstageEmbedding

root_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(root_path, 'data')


@click.command()
@click.option('--config', type=click.Path(exists=True), default=os.path.join(root_path, 'config',
                                                                             'embedding_benchmark.yaml'))
@click.option('--qa_data_path', type=click.Path(exists=True), default=os.path.join(data_path, 'qa_v4.parquet'))
@click.option('--corpus_data_path', type=click.Path(exists=True),
              default=os.path.join(data_path, 'ocr_corpus_v3.parquet'))
@click.option('--project_dir', type=click.Path(exists=False), default=os.path.join(root_path, 'benchmark'))
def main(config, qa_data_path, corpus_data_path, project_dir):
    load_dotenv()
    autorag.embedding_models['local_model'] = autorag.LazyInit(HuggingFaceEmbedding,
                                                                        model_name="/data/MODELS/240708") # full path of the local model

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    evaluator = Evaluator(qa_data_path, corpus_data_path, project_dir=project_dir)
    evaluator.start_trial(config)


if __name__ == '__main__':
    main()
