# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, confusion_matrix, make_scorer
# for model serialization
import joblib
# for creating a folder and parsing args
import os
import argparse
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description="Train XGBoost Classifier with Custom Model Version")
parser.add_argument("--model_version", type=str, default="model_0", help="Name of the model version (e.g., model_0, model_1, model_2)")
args = parser.parse_args()

# --- Configuration ---
HF_REPO_ID = "lomface/Tourism-Package-Prediction"
HF_MODEL_REPO_ID = "lomface/tourism-package-prediction-model"
TARGET_VARIABLE = 'ProdTaken'
LOCAL_OUTPUT_DIR = "tourism_project/model_building/joblibs"
RANDOM_STATE = 42
MODEL_VERSION = args.model_version

# --- Ensure output directory exists ---
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# --- Initialize Hugging Face API client ---
hf_api = HfApi(token=os.getenv("HF_TOKEN"))

# --- Create Hugging Face Model Repository if it doesn't exist ---
try:
    hf_api.repo_info(repo_id=HF_MODEL_REPO_ID, repo_type="model")
    print(f"Hugging Face model space '{HF_MODEL_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Hugging Face model space '{HF_MODEL_REPO_ID}' not found. Creating new space...")
    create_repo(repo_id=HF_MODEL_REPO_ID, repo_type="model", private=False)
    print(f"Hugging Face model space '{HF_MODEL_REPO_ID}' created.")

# --- Load Data from Hugging Face ---
print(f"Loading split data from Hugging Face repository: {HF_REPO_ID}")

X_train = pd.read_csv(hf_api.hf_hub_download(repo_id=HF_REPO_ID, filename="X_train.csv", repo_type="dataset"))
X_val = pd.read_csv(hf_api.hf_hub_download(repo_id=HF_REPO_ID, filename="X_val.csv", repo_type="dataset"))
X_test = pd.read_csv(hf_api.hf_hub_download(repo_id=HF_REPO_ID, filename="X_test.csv", repo_type="dataset"))
y_train = pd.read_csv(hf_api.hf_hub_download(repo_id=HF_REPO_ID, filename="y_train.csv", repo_type="dataset")).squeeze()
y_val = pd.read_csv(hf_api.hf_hub_download(repo_id=HF_REPO_ID, filename="y_val.csv", repo_type="dataset")).squeeze()
y_test = pd.read_csv(hf_api.hf_hub_download(repo_id=HF_REPO_ID, filename="y_test.csv", repo_type="dataset")).squeeze()

# --- Identify numerical features for scaling ---
numerical_features_for_scaling = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]
numerical_features_for_scaling = [col for col in numerical_features_for_scaling if col in X_train.columns]

# --- Preprocessing: Scaling numerical features ---
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features_for_scaling] = scaler.fit_transform(X_train[numerical_features_for_scaling])
X_val_scaled[numerical_features_for_scaling] = scaler.transform(X_val[numerical_features_for_scaling])
X_test_scaled[numerical_features_for_scaling] = scaler.transform(X_test[numerical_features_for_scaling])

# --- Handle Class Imbalance with scale_pos_weight ---
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight = neg_count / pos_count

# --- Configure Hyperparameters based on Model Version ---
if MODEL_VERSION == "model_0":
    # Baseline search
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }
elif MODEL_VERSION == "model_2":
    # Regularization & Imbalance-Focused Model
    param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.03, 0.05],
        'max_depth': [3, 4],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3]
    }
else:
    # Wider search space (model_1)
    param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                              random_state=RANDOM_STATE, scale_pos_weight=scale_pos_weight)

f1_scorer = make_scorer(f1_score, pos_label=1)

print(f"\nStarting GridSearchCV for {MODEL_VERSION}...")
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring=f1_scorer, cv=3, verbose=1, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)

best_xgb_model = grid_search.best_estimator_
print(f"\nBest XGBoost parameters found: {grid_search.best_params_}")

# --- Model Evaluation ---
y_pred = best_xgb_model.predict(X_test_scaled)
print(f"\n--- {MODEL_VERSION} Evaluation on Test Set ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score (Positive Class): {f1_score(y_test, y_pred, pos_label=1):.4f}")

# --- Save and Upload Fitted Scaler ---
scaler_filename = "scaler.joblib"
scaler_filepath = os.path.join(LOCAL_OUTPUT_DIR, scaler_filename)
joblib.dump(scaler, scaler_filepath)

hf_api.upload_file(
    path_or_fileobj=scaler_filepath,
    path_in_repo=scaler_filename,
    repo_id=HF_MODEL_REPO_ID,
    repo_type="model",
    commit_message=f"Upload fitted StandardScaler for {MODEL_VERSION}"
)

# --- Save and Upload Model ---
model_filename = f"{MODEL_VERSION}.joblib"
model_filepath = os.path.join(LOCAL_OUTPUT_DIR, model_filename)
joblib.dump(best_xgb_model, model_filepath)
print(f"Saved {MODEL_VERSION} locally to {model_filepath}")

hf_api.upload_file(
    path_or_fileobj=model_filepath,
    path_in_repo=model_filename,
    repo_id=HF_MODEL_REPO_ID,
    repo_type="model",
    commit_message=f"Upload trained XGBoost model version: {model_filename}"
)
print("Artifact upload to Hugging Face complete!")
