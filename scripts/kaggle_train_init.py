import os
import json
import subprocess
import shutil

def init_kaggle_dataset():
    print("🚀 Synchronizing Kaggle Dataset (V3.2) for Realization Engine...")
    staging_dir = "kaggle_staging"
    if os.path.exists(staging_dir): shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    # Copy structured data directories
    for sub in ['realizations', 'scenarios', 'framework', 'results']:
        src = os.path.join("data", sub)
        if os.path.exists(src):
            shutil.copytree(src, os.path.join(staging_dir, sub))

    # Copy core directory as a directory for Kaggle to zip
    if os.path.exists("core"):
        shutil.copytree("core", os.path.join(staging_dir, "core"))

    # Copy metadata files specifically
    for f in ["dataset-metadata.json"]:
        src = os.path.join("data", f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(staging_dir, f))

    # Dataset metadata check/update
    metadata_path = os.path.join(staging_dir, "dataset-metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "title": "Realization Engine Knowledge Base",
            "id": "djangolimited/realization-engine-data",
            "licenses": [{"name": "CC0-1.0"}]
        }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Staging area prepared at kaggle_staging/")

    # Check if kaggle is installed
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        print("🎉 Kaggle CLI detected. Attempting to sync...")
        # Use try-except for the version update
        try:
            print("Attempting to update version...")
            subprocess.run(["kaggle", "datasets", "version", "-p", staging_dir, "-m", "Robust UQS V3.2 with hierarchical datasets", "--dir-mode", "zip"], check=True)
            print("🎉 Dataset synchronized successfully!")
        except subprocess.CalledProcessError:
            print("Update failed, attempting to create...")
            subprocess.run(["kaggle", "datasets", "create", "-p", staging_dir, "--dir-mode", "zip"], check=True)
            print("🎉 Dataset created successfully!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Kaggle CLI not found or credentials missing. Please run the following command manually:")
        print(f"kaggle datasets version -p {staging_dir} -m \"Robust UQS V3.2 with hierarchical datasets\" --dir-mode zip")

if __name__ == "__main__":
    init_kaggle_dataset()
