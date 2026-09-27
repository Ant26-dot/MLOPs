# This script (`prep_2.py`) is designed for the MLOps pipeline.
# It is a standalone Python script intended to be executed by automation tools (e.g., GitHub Actions)
# to ensure consistent data preprocessing for model training and deployment.
# It should not rely on the interactive state or global variables defined directly within the Colab notebook.
# For interactive development and experimentation within the notebook, global constants are defined
# in a separate notebook cell (e.g., `35ea4d99`) to make variables readily available to subsequent cells
# without re-executing the full pipeline script each time.

import pandas as pd
import os
from huggingface_hub import hf_hub_download, HfApi

# Ensure that all columns will be shown
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Re-load the dataset and apply the same preprocessing steps as in prep_2.py
# This is necessary to correctly identify numerical columns after one-hot encoding
df = pd.read_csv(hf_hub_download(repo_id="lomface/Tourism-Package-Prediction", filename="tourism.csv", repo_type="dataset"))

print("Original dataset shape:", df.shape)
print("Original missing values:\n", df.isna().sum())

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
    print("\nDropped 'Unnamed: 0' column.")

# Drop CustomerID as it's an identifier and not a feature for the model
if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])
    print("Dropped 'CustomerID' column.")

categorical_cols = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nApplied one-hot encoding to categorical features.")

# Define the target variable
target_variable = 'ProdTaken'
print(f"\nTarget Variable: {target_variable}")

# Identify numerical features (excluding the target)
all_columns = df.columns.tolist()
numerical_features = [col for col in all_columns if df[col].dtype in ['int64', 'float64', 'uint8'] and col != target_variable]

print(f"\nNumerical Features: {numerical_features}")

print("\nDataset shape after cleaning and encodingに参加しました:", df.shape)
print("\nFinal data types:\n", df.info())
print("\nFirst 5 rows of cleaned and encoded dataset:\n", df.head())

# --- Save and Upload Processed DataFrame ---
output_dir = "tourism_project/data"
os.makedirs(output_dir, exist_ok=True)
output_filepath = os.path.join(output_dir, "tourism_processed.csv")
df.to_csv(output_filepath, index=False)
print(f"\nProcessed DataFrame saved to {output_filepath}")

# Upload the processed CSV file to Hugging Face
# This script is designed to be run in an environment where HF_TOKEN is available as an environment variable.
HF_API_CLIENT = HfApi(token=os.getenv("HF_TOKEN"))
HF_REPO_ID = "lomface/Tourism-Package-Prediction" # Use the project's designated repo ID
hf_dataset_filename_processed = "tourism_processed.csv"

print(f"\nUploading {output_filepath} to Hugging Face repository {HF_REPO_ID} as {hf_dataset_filename_processed}...")

HF_API_CLIENT.upload_file(
    path_or_fileobj=output_filepath,
    path_in_repo=hf_dataset_filename_processed,
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    commit_message=f"Upload processed tourism data: {hf_dataset_filename_processed}"
)

print("Upload to Hugging Face complete!")
