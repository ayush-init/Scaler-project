import os
import getpass
from huggingface_hub import HfApi

def deploy():
    print("=" * 60)
    print("🚀 DB-SURGEON HUGGINGFACE DEPLOYER")
    print("=" * 60)
    print("You can get a Write Token from: https://huggingface.co/settings/tokens")
    print()
    
    token = getpass.getpass("Enter your HuggingFace Write Token (it will be hidden as you type): ")
    if not token.strip():
        print("Token is required!")
        return

    repo_id = "ayush0211/db-surgeon"
    api = HfApi(token=token.strip())
    
    print(f"\nAuthenticating and connecting to {repo_id}...")
    try:
        api.whoami()
    except Exception as e:
        print(f"❌ Authentication failed! Ensure the token has 'Write' permissions.\nError: {e}")
        return

    base_dir = r"e:\Scaler\db_surgeon"
    
    print("\n[1/4] Uploading app.py...")
    api.upload_file(
        path_or_fileobj=os.path.join(base_dir, "hf_space", "app.py"),
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space"
    )
    
    print("[2/4] Uploading requirements.txt...")
    api.upload_file(
        path_or_fileobj=os.path.join(base_dir, "hf_space", "requirements.txt"),
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space"
    )
    
    print("[3/4] Uploading README.md (Space configuration)...")
    api.upload_file(
        path_or_fileobj=os.path.join(base_dir, "hf_space", "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space"
    )
    
    print("[4/4] Uploading db_surgeon core package files...")
    api.upload_folder(
        folder_path=base_dir,
        path_in_repo="db_surgeon",
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=["models.py", "client.py", "__init__.py", "server/*.py", "training/*.py", "examples/*.py"]
    )
    
    print("\n" + "=" * 60)
    print("✅ DEPLOYMENT SUCCESSFUL!")
    print(f"The app, requirements, and db_surgeon package have been pushed to HF!")
    print(f"View it live at: https://huggingface.co/spaces/{repo_id}")
    print("=" * 60)

if __name__ == "__main__":
    deploy()
