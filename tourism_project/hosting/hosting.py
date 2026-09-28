from huggingface_hub import HfApi
import os

# --- Configuration ---
# The target Hugging Face Space repository
HF_SPACE_REPO_ID = "lomface/Tourism-Package-Prediction"
LOCAL_DEPLOY_DIR = "tourism_project/deployment"

# Initialize the Hugging Face API client using the loaded environment token
api = HfApi(token=os.getenv("HF_TOKEN"))

print(f"Uploading deployment folder contents from '{LOCAL_DEPLOY_DIR}' to Hugging Face Space '{HF_SPACE_REPO_ID}'...")
try:
    api.upload_folder(
        folder_path=LOCAL_DEPLOY_DIR,
        repo_id=HF_SPACE_REPO_ID,
        repo_type="space",
        path_in_repo=""
    )
    print("Hosting upload to Hugging Face Spaces complete!")
except Exception as e:
    print(f"Error uploading to Hugging Face Spaces: {e}")
