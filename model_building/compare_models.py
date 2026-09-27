import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from huggingface_hub import HfApi

# --- Configuration ---
HF_REPO_ID = "lomface/Tourism-Package-Prediction"
HF_MODEL_REPO_ID = "lomface/tourism-package-prediction-model"
TARGET_VARIABLE = 'ProdTaken'
LOCAL_MODEL_DIR = "tourism_project/model_building/joblibs"
MODELS = ["model_0", "model_1", "model_2"]

# Initialize HF Client
hf_api = HfApi(token=os.getenv("HF_TOKEN"))

print("Loading split data for champion evaluation...")
X_test = pd.read_csv(hf_api.hf_hub_download(repo_id=HF_REPO_ID, filename="X_test.csv", repo_type="dataset"))
y_test = pd.read_csv(hf_api.hf_hub_download(repo_id=HF_REPO_ID, filename="y_test.csv", repo_type="dataset")).squeeze()

# Load the scaler fitted during training
scaler_path = hf_api.hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename="scaler.joblib", repo_type="model")
scaler = joblib.load(scaler_path)

# Scaling numerical features for testing
numerical_features_for_scaling = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]
numerical_features_for_scaling = [col for col in numerical_features_for_scaling if col in X_test.columns]

X_test_scaled = X_test.copy()
X_test_scaled[numerical_features_for_scaling] = scaler.transform(X_test[numerical_features_for_scaling])

results = {}
best_model_name = None
best_f1 = -1.0

print("\n=== Evaluating All Model Versions ===")
for model_name in MODELS:
    model_filename = f"{model_name}.joblib"
    try:
        # Download from model space
        model_path = hf_api.hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=model_filename, repo_type="model")
        model = joblib.load(model_path)

        # Generate predictions
        y_pred = model.predict(X_test_scaled)

        # Compute metrics
        f1 = f1_score(y_test, y_pred, pos_label=1)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1)

        results[model_name] = {"F1": f1, "Accuracy": acc, "Recall": rec, "Precision": prec}
        print(f"{model_name} -> F1: {f1:.4f} | Accuracy: {acc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f}")

        # Champion criteria (Max positive F1-score)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name

    except Exception as e:
        print(f"Could not evaluate {model_name}: {e}")

print(f"\n Champion Model Selected: {best_model_name} with F1-Score: {best_f1:.4f}")

# --- Copy champion model and rename it as best_model.joblib ---
winning_model_local_path = os.path.join(LOCAL_MODEL_DIR, f"{best_model_name}.joblib")
production_model_local_path = os.path.join(LOCAL_MODEL_DIR, "best_model.joblib")

# Load winning model from local artifact directory and serialize to production name
winning_model = joblib.load(winning_model_local_path)
joblib.dump(winning_model, production_model_local_path)
print(f"Successfully saved {best_model_name} as local production target: {production_model_local_path}")

# --- Upload champion to Hugging Face model repository as best_model.joblib ---
hf_api.upload_file(
    path_or_fileobj=production_model_local_path,
    path_in_repo="best_model.joblib",
    repo_id=HF_MODEL_REPO_ID,
    repo_type="model",
    commit_message=f"Promote {best_model_name} as production best_model.joblib"
)
print("Uploaded production best_model.joblib to Hugging Face Model Hub!")
