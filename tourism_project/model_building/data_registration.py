from datasets import Dataset
from huggingface_hub import HfApi
import pandas as pd
import os

# Get Hugging Face token from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# Check if token was successfully retrieved
if not HF_TOKEN:
    print("Cannot proceed without a valid Hugging Face token.")
    print("Please ensure 'HF_TOKEN' is set as an environment variable.")
    raise ValueError("Hugging Face token is required but not found or accessible.")
else:
    print(f"HF_TOKEN retrieved from environment variables: {HF_TOKEN[:5]}...")

# Define paths and repository details
DATA_PATH = "tourism_project/data/tourism.csv"
REPO_ID = os.environ.get("HF_REPO_ID", "nareshpaib/tourism_dataset") # Replace with your Hugging Face username and desired repo name

# Load the dataset
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded data from {DATA_PATH}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found. Please ensure tourism.csv is uploaded to the 'tourism_project/data' folder.")
    raise FileNotFoundError(f"{DATA_PATH} not found.")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Convert pandas DataFrame to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df)

# Authenticate with Hugging Face Hub
api = HfApi(token=HF_TOKEN)

# Create a new dataset repository if it doesn't exist
try:
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"Hugging Face dataset repository '{REPO_ID}' created or already exists.")
except Exception as e:
    print(f"Error creating Hugging Face repository: {e}")
    raise

# Push the dataset to Hugging Face Hub
try:
    hf_dataset.push_to_hub(REPO_ID, private=True)
    print(f"Dataset successfully pushed to Hugging Face Hub: https://huggingface.co/datasets/{REPO_ID}")
except ValueError as e:
    if "Features of the new split don't match" in str(e):
        print("Schema mismatch detected. Deleting existing dataset and pushing new one...")
        try:
            api.delete_repo(repo_id=REPO_ID, repo_type="dataset")
            print(f"Deleted existing dataset: {REPO_ID}")
        except Exception as delete_error:
            print(f"Could not delete repo (may not exist): {delete_error}")
        
        # Recreate the repo and push
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        hf_dataset.push_to_hub(REPO_ID, private=True)
        print(f"Dataset successfully pushed to Hugging Face Hub: https://huggingface.co/datasets/{REPO_ID}")
    else:
        raise
except Exception as e:
    print(f"Error pushing dataset to Hugging Face Hub: {e}")
    raise
