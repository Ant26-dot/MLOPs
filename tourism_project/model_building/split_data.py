
import pandas as pd
import os
from huggingface_hub import hf_hub_download, HfApi
from sklearn.model_selection import train_test_split

# --- Configuration ---
HF_REPO_ID = "lomface/Tourism-Package-Prediction"
HF_PROCESSED_FILENAME = "tourism_processed.csv"
TARGET_VARIABLE = 'ProdTaken'
TEST_SIZE_RATIO = 0.2    # 20% for test
VAL_SIZE_RATIO = 0.25  # 25% of the remaining (train+val) for validation
RANDOM_STATE = 42
LOCAL_OUTPUT_DIR = "tourism_project/data"

# Ensure output directory exists
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

print(f"Loading processed data from Hugging Face: {HF_PROCESSED_FILENAME}")
df_processed = pd.read_csv(hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_PROCESSED_FILENAME,
    repo_type="dataset"
))

print(f"Loaded processed data with shape: {df_processed.shape}")

# Separate features (X) and target (y)
X = df_processed.drop(columns=[TARGET_VARIABLE])
y = df_processed[TARGET_VARIABLE]

# First, split into training+validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE_RATIO, random_state=RANDOM_STATE, stratify=y
)

# Then, split the training+validation set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VAL_SIZE_RATIO, random_state=RANDOM_STATE, stratify=y_train_val
)

print(f"Data split completed:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_val shape: {X_val.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_val shape: {y_val.shape}")
print(f"  y_test shape: {y_test.shape}")

# --- Save split data locally ---
X_train_path = os.path.join(LOCAL_OUTPUT_DIR, "X_train.csv")
X_val_path = os.path.join(LOCAL_OUTPUT_DIR, "X_val.csv")
X_test_path = os.path.join(LOCAL_OUTPUT_DIR, "X_test.csv")
y_train_path = os.path.join(LOCAL_OUTPUT_DIR, "y_train.csv")
y_val_path = os.path.join(LOCAL_OUTPUT_DIR, "y_val.csv")
y_test_path = os.path.join(LOCAL_OUTPUT_DIR, "y_test.csv")

X_train.to_csv(X_train_path, index=False)
X_val.to_csv(X_val_path, index=False)
X_test.to_csv(X_test_path, index=False)
y_train.to_csv(y_train_path, index=False)
y_val.to_csv(y_val_path, index=False)
y_test.to_csv(y_test_path, index=False)

print(f"Split data saved locally to {LOCAL_OUTPUT_DIR}/")

# --- Upload split data to Hugging Face ---
hf_api = HfApi(token=os.getenv("HF_TOKEN"))

files_to_upload = {
    X_train_path: "X_train.csv",
    X_val_path: "X_val.csv",
    X_test_path: "X_test.csv",
    y_train_path: "y_train.csv",
    y_val_path: "y_val.csv",
    y_test_path: "y_test.csv",
}

print(f"Uploading split data to Hugging Face repository {HF_REPO_ID}...")

for local_path, hf_filename in files_to_upload.items():
    hf_api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=hf_filename,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Upload split data: {hf_filename}"
    )
    print(f"  Uploaded {hf_filename}")

print("Upload of split data to Hugging Face complete!")
