"""
upload.py — Push hf_survival to HuggingFace Space
Usage: python upload.py [message]
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "muqing-research/nsclc-survival"
FILES   = ["survival_app.py", "survival_core.py", "app.py", "requirements.txt",
           "Dockerfile", "world.geojson", "tcga_luad_app_bundle.pkl"]

msg = sys.argv[1] if len(sys.argv) > 1 else "Update"
api = HfApi()
folder = Path(__file__).parent

api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)
print(f"  repo ready: {REPO_ID}")

for f in FILES:
    api.upload_file(
        path_or_fileobj=str(folder / f),
        path_in_repo=f,
        repo_id=REPO_ID,
        repo_type="space",
        commit_message=msg,
    )
    print(f"  uploaded: {f}")

print(f"\nDone — https://huggingface.co/spaces/{REPO_ID}")
