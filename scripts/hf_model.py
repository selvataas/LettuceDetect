from huggingface_hub import create_repo, upload_folder

repo_id = "selvatas/lettucedect-210m-eurobert-tr-v1"
local_path = "/home/stas/lettucedetect/output/hallucination_detector"

create_repo(repo_id, repo_type="model", exist_ok=True)
upload_folder(folder_path=local_path, repo_id=repo_id, repo_type="model")
