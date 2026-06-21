import os
from huggingface_hub import HfApi

# Get environment variables
HF_TOKEN = os.environ.get('HF_TOKEN')
SPACE_REPO_ID = os.environ.get('HF_SPACE_REPO_ID')
MODEL_REPO_ID = os.environ.get('HF_MODEL_REPO_ID')

print(f'Hugging Face Space repository: {SPACE_REPO_ID}')
print(f'Hugging Face Model repository: {MODEL_REPO_ID}')

# Initialize API
api = HfApi(token=HF_TOKEN)

# Create or update space repo
try:
    api.create_repo(repo_id=SPACE_REPO_ID, repo_type='space', space_sdk='docker', exist_ok=True)
    print(f'Space repo created or exists: {SPACE_REPO_ID}')
except Exception as e:
    print(f'Error creating space repo: {e}')
    raise

# Define deployment files
deployment_files = [
    'tourism_project/deployment/Dockerfile',
    'tourism_project/deployment/app.py',
    'tourism_project/requirements.txt'
]

# Upload files
print(f'Uploading deployment files to {SPACE_REPO_ID}...')
for file_path in deployment_files:
    if os.path.exists(file_path):
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),
                repo_id=SPACE_REPO_ID,
                repo_type='space'
            )
            print(f'Uploaded {os.path.basename(file_path)}')
        except Exception as e:
            print(f'Error uploading {file_path}: {e}')
            raise
    else:
        print(f'Warning: File not found: {file_path}')

print(f'Deployment files pushed to: https://huggingface.co/spaces/{SPACE_REPO_ID}')
