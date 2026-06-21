from datasets import load_dataset
from huggingface_hub import HfApi
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib # To save/load models

# Set MLflow tracking URI to a local directory for now
mlflow.set_tracking_uri("file:///content/mlruns")
mlflow.set_experiment("tourism_package_prediction")

# Get Hugging Face token from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# Check if token was successfully retrieved
if not HF_TOKEN:
    print("Cannot proceed without a valid Hugging Face token.")
    print("Please ensure 'HF_TOKEN' is set as an environment variable.")
    raise ValueError("Hugging Face token is required but not found or accessible.")
else:
    print(f"HF_TOKEN retrieved from environment variables: {HF_TOKEN[:5]}...")

# Define Hugging Face repository details for dataset and model
DATA_REPO_ID = os.environ.get("HF_REPO_ID", "nareshpaib/tourism_dataset")
MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID", "nareshpaib/tourism_classifier") # New repo for model

# Load the processed train and test datasets from Hugging Face Hub
try:
    train_dataset = load_dataset(DATA_REPO_ID, split='train')
    test_dataset = load_dataset(DATA_REPO_ID, split='test')

    df_train = train_dataset.to_pandas()
    df_test = test_dataset.to_pandas()
    print(f"Successfully loaded train data from Hugging Face Hub. Shape: {df_train.shape}")
    print(f"Successfully loaded test data from Hugging Face Hub. Shape: {df_test.shape}")
except Exception as e:
    print(f"Error loading dataset from Hugging Face Hub: {e}")
    raise

# Separate features (X) and target (y)
X_train = df_train.drop('ProdTaken', axis=1)
y_train = df_train['ProdTaken']
X_test = df_test.drop('ProdTaken', axis=1)
y_test = df_test['ProdTaken']

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['number']).columns

# Apply one-hot encoding to categorical features
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns - crucial for consistent feature sets after one-hot encoding
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols] # Ensure the order of columns is the same

print(f"X_train shape after one-hot encoding and alignment: {X_train.shape}")
print(f"X_test shape after one-hot encoding and alignment: {X_test.shape}")

# --- Model Training and Experimentation Tracking ---

# Initialize MLflow run
with mlflow.start_run():
    # Define the model and parameter grid for GridSearchCV
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    print("Starting GridSearchCV for RandomForestClassifier...")
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best parameters found: {best_params}")

    # Log parameters to MLflow
    mlflow.log_params(best_params)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Model Performance on Test Set:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC AUC Score: {roc_auc:.4f}")

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc_score", roc_auc)

    # Log the best model to MLflow
    mlflow.sklearn.log_model(best_model, "best_random_forest_model")
    print("Best model logged to MLflow.")

    # --- Register the best model to Hugging Face Model Hub ---

    # Create a local directory for the model artifact
    model_path = "tourism_project/model_artifacts/random_forest_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Best model saved locally at: {model_path}")

    api = HfApi(token=HF_TOKEN)

    # Create a model repository if it doesn't exist
    try:
        api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)
        print(f"Hugging Face model repository '{MODEL_REPO_ID}' created or already exists.")
    except Exception as e:
        print(f"Error creating Hugging Face model repository: {e}")
        raise

    # Upload the model artifact
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="random_forest_model.joblib",
            repo_id=MODEL_REPO_ID,
            repo_type="model",
        )
        print(f"Model successfully pushed to Hugging Face Model Hub: https://huggingface.co/{MODEL_REPO_ID}")
    except Exception as e:
        print(f"Error pushing model to Hugging Face Model Hub: {e}")
        raise
