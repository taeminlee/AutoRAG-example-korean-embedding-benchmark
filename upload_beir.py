from huggingface_hub import HfApi
api = HfApi()

repo_id = "nlpai-lab/markers_bm"

api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, )

# raw_file upload
api.upload_folder(
    folder_path="./markers_bm",
    repo_id=repo_id,
    repo_type="dataset",
)