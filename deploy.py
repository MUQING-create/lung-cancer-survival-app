"""
deploy.py -- Deploy survival app to shinyapps.io
Usage: python deploy.py
"""
import subprocess
import sys
from pathlib import Path

ACCOUNT = "medictio"
TITLE   = "nsclc-survival"
DIR     = str(Path(__file__).parent)
EXCLUDE = ["Dockerfile", "upload.py", "deploy.py"]

cmd = (
    ["rsconnect", "deploy", "shiny", DIR, "--name", ACCOUNT, "--title", TITLE]
    + [arg for f in EXCLUDE for arg in ("--exclude", f)]
)

print(f"Deploying '{TITLE}' to shinyapps.io ({ACCOUNT})...")
result = subprocess.run(cmd)

if result.returncode == 0:
    print(f"\nDone -- https://{ACCOUNT}.shinyapps.io/{TITLE}/")
else:
    print("\nDeployment failed.", file=sys.stderr)
    sys.exit(result.returncode)
