from datasets import Dataset
from huggingface_hub import HfApi, create_repo
import pandas as pd
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# Hugging Faceリポジトリの設定
# あなたのHugging Faceユーザー名/組織名とデータセットリポジトリ名に置き換えてください
dataset_repo_id = os.getenv("DATASET_REPO_ID")
hf_token = os.getenv("HF_TOKEN")

if hf_token is None:
    print(
        "Hugging Face token (HF_TOKEN) not found in environment variables or .env file."
    )
    print(
        "Please set it before running the script, or replace 'os.environ.get(\"HF_TOKEN\")' with your actual token."
    )
    exit()

api = HfApi(token=hf_token)

# データセットリポジトリが存在しない場合は作成
try:
    create_repo(
        repo_id=dataset_repo_id, repo_type="dataset", exist_ok=True, token=hf_token
    )
    print(f"Dataset repository '{dataset_repo_id}' created or already exists.")
except Exception as e:
    print(f"Error creating/checking dataset repository: {e}")
    exit()

# Parquetファイルの読み込み
data_file_path = "./data/index_dataset.parquet"
try:
    df = pd.read_parquet(data_file_path)
    print(f"Successfully loaded data from {data_file_path}")
except Exception as e:
    print(f"Error loading parquet file: {e}")
    exit()

# Pandas DataFrameをHugging Face Datasetに変換
dataset = Dataset.from_pandas(df)

# データセットをHugging Face Hubにプッシュ
try:
    dataset.push_to_hub(dataset_repo_id, token=hf_token)
    print(f"Dataset successfully pushed to Hugging Face Hub: {dataset_repo_id}")
except Exception as e:
    print(f"Error pushing dataset to Hugging Face Hub: {e}")

print("Dataset upload process completed.")
