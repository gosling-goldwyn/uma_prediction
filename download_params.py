import os
import argparse
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からトークンを読み込む
HF_TOKEN = os.environ.get("HF_TOKEN")


def download_all_params_from_hf(repo_id, output_dir, token):
    """
    Hugging Face Hubからすべての最適化されたパラメータファイルをダウンロードする。
    """
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        print("Please set it to your Hugging Face access token.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Define the model types for which parameters are expected
    model_types = ["rf", "lgbm", "cnn"]  # Assuming these are the only types

    for model_type in model_types:
        param_file_name = f"best_params_{model_type}.json"
        path_in_repo = f"params/{param_file_name}"

        print(f"Attempting to download '{path_in_repo}' from '{repo_id}'...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                token=token,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            print(f"Successfully downloaded to: {downloaded_path}")
        except Exception as e:
            print(f"Warning: Could not download {path_in_repo} from {repo_id}: {e}")
            print(
                "This might mean the file does not exist or your token has insufficient access."
            )

    print("Parameter download process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all optimized parameters from Hugging Face Hub."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="params",
        help="Local directory to save the downloaded parameters.",
    )
    args = parser.parse_args()

    repo_id = os.getenv("REPO_ID")
    if not repo_id:
        print("Error: REPO_ID environment variable not set.")
        print(
            "Please set it to your Hugging Face repository ID (e.g., your-username/uma-prediction-models)."
        )
        exit(1)

    download_all_params_from_hf(repo_id, args.output_dir, HF_TOKEN)
