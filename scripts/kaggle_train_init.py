import os
import json
import subprocess
import shutil

def init_kaggle_dataset():
    print("🚀 Synchronizing Kaggle Dataset (V3.1) for Realization Engine...")
    staging_dir = "kaggle_staging"
    if os.path.exists(staging_dir): shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    # Copy all data files including the new ones
    for f in os.listdir("data"):
        src = os.path.join("data", f)
        if os.path.isfile(src): shutil.copy(src, staging_dir)

    # Copy core directory as is
    shutil.copytree("core", os.path.join(staging_dir, "core"))

    # Dataset metadata
    metadata = {"title": "Realization Engine Knowledge Base", "id": "djangolimited/realization-engine-data", "licenses": [{"name": "CC0-1.0"}]}
    with open(os.path.join(staging_dir, "dataset-metadata.json"), "w") as f: json.dump(metadata, f, indent=2)

    try:
        subprocess.run(["kaggle", "datasets", "version", "-p", staging_dir, "-m", "Evolved UQS V3.1 with emergents and new optimized prompts", "--dir-mode", "zip"], check=True)
        print("🎉 Dataset synchronized successfully!")
    except subprocess.CalledProcessError:
        subprocess.run(["kaggle", "datasets", "create", "-p", staging_dir, "--dir-mode", "zip"], check=True)
        print("🎉 Dataset created successfully!")

if __name__ == "__main__":
    init_kaggle_dataset()
