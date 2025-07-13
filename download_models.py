from huggingface_hub import HfApi, hf_hub_download
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get repository ID from environment variables
repo_id = os.getenv("REPO_ID")
if not repo_id:
    print("Error: REPO_ID not found in environment variables or .env file.")
    exit()

# Initialize Hugging Face API
api = HfApi()

try:
    # Get the list of files from the repository
    print(f"Fetching file list from repository: {repo_id}...")
    repo_files = api.list_repo_files(repo_id)
    print("File list fetched successfully.")

    # Define the local directory to save models
    local_dir = "models"
    os.makedirs(local_dir, exist_ok=True)

    # Download each file
    for filename in repo_files:
        # Skip directories or other non-model files if necessary (e.g., .gitattributes)
        if filename.startswith('.'):
            print(f"Skipping hidden file: {filename}")
            continue

        print(f"Downloading {filename} to {local_dir}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # To avoid symlinks
            force_download=True,  # Force download to overwrite existing files
        )

    print("\nAll models downloaded successfully!")

except Exception as e:
    print(f"An error occurred: {e}")

