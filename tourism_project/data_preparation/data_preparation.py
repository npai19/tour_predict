from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Get Hugging Face token from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# Check if token was successfully retrieved
if not HF_TOKEN:
    print("Cannot proceed without a valid Hugging Face token.")
    print("Please ensure 'HF_TOKEN' is set as an environment variable.")
    raise ValueError("Hugging Face token is required but not found or accessible.")
else:
    print(f"HF_TOKEN retrieved from environment variables: {HF_TOKEN[:5]}...")

# Define Hugging Face repository details
REPO_ID = os.environ.get("HF_REPO_ID", "nareshpaib/tourism_dataset")

# Load the dataset from Hugging Face Hub
try:
    hf_dataset = load_dataset(REPO_ID)
    df = hf_dataset['train'].to_pandas()
    print(f"Successfully loaded data from Hugging Face Hub. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset from Hugging Face Hub: {e}")
    raise

# --- Data Cleaning and Preprocessing ---

# Drop 'Unnamed: 0' column if it exists, as it's often an artifact of saving/loading DataFrames
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
    print("Dropped 'Unnamed: 0' column.")

# Drop CustomerID as it's not a feature for modeling
if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])
    print("Dropped 'CustomerID' column.")

# Handle missing values
# Separate numerical and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Impute numerical missing values with the mean
if numeric_cols:
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    print("Imputed numerical missing values with the mean.")

# Impute categorical missing values with the most frequent value (mode)
if categorical_cols:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    print("Imputed categorical missing values with the mode.")

# Verify no missing values remain
print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# --- Split the data into training and testing sets ---

# Define features (X) and target (y)
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Recombine for easier pushing to Hugging Face
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

print(f"Train set shape: {df_train.shape}")
print(f"Test set shape: {df_test.shape}")

# --- Save processed datasets locally ---
PROCESSED_DATA_PATH = "tourism_project/processed_data"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
df_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "train.csv"), index=False)
df_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "test.csv"), index=False)
print(f"Processed train and test datasets saved locally to {PROCESSED_DATA_PATH}")

# Convert pandas DataFrames back to Hugging Face Datasets
hf_train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))
hf_test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))

# Authenticate with Hugging Face Hub
api = HfApi(token=HF_TOKEN)

# Push the processed datasets to Hugging Face Hub
try:
    # Create or update the 'train' split
    hf_train_dataset.push_to_hub(REPO_ID, split="train", commit_message="Add processed train split")
    print(f"Train dataset successfully pushed to Hugging Face Hub: https://huggingface.co/datasets/{REPO_ID}/viewer/default/train")

    # Create or update the 'test' split
    hf_test_dataset.push_to_hub(REPO_ID, split="test", commit_message="Add processed test split")
    print(f"Test dataset successfully pushed to Hugging Face Hub: https://huggingface.co/datasets/{REPO_ID}/viewer/default/test")

except Exception as e:
    print(f"Error pushing processed datasets to Hugging Face Hub: {e}")
    raise
