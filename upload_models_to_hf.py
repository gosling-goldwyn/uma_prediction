from huggingface_hub import HfApi, create_repo
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# Hugging Faceリポジトリの設定
repo_id = os.getenv("REPO_ID")
# 環境変数からHugging Faceトークンを読み込む
hf_token = os.getenv("HF_TOKEN")

if hf_token is None:
    print(
        "Hugging Face token (HF_TOKEN) not found in environment variables or .env file."
    )
    print(
        "Please ensure .env file exists and contains HF_TOKEN, or set it as an environment variable."
    )
    exit()

api = HfApi(token=hf_token)

# リポジトリが存在しない場合は作成
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=hf_token)
    print(f"Repository '{repo_id}' created or already exists.")
except Exception as e:
    print(f"Error creating/checking repository: {e}")
    exit()

# アップロードするファイルのパス
model_dir = "./models"
files_to_upload = []
for root, _, files in os.walk(model_dir):
    for file in files:
        files_to_upload.append(os.path.join(root, file))

print(f"Found {len(files_to_upload)} files to upload from '{model_dir}'.")

# ファイルをHugging Faceにアップロード
for file_path in files_to_upload:
    try:
        # リポジトリ内のパスは、model_dirからの相対パスにする
        repo_file_path = os.path.relpath(file_path, model_dir)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=repo_file_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
        )
        print(f"Uploaded {file_path} to {repo_id}/{repo_file_path}")
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")

print("Model upload process completed.")
