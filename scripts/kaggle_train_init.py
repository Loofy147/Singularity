import os
import json
import subprocess

def init_kaggle_dataset():
    print("🚀 Initializing Kaggle Dataset for Realization Engine...")

    # Create metadata for the dataset
    dataset_dir = "data"
    metadata = {
        "title": "Realization Engine Knowledge Base",
        "id": "djangolimited/realization-engine-data",
        "licenses": [{"name": "CC0-1.0"}]
    }

    with open(os.path.join(dataset_dir, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Created dataset-metadata.json in data/")

    # Try to create the dataset on Kaggle
    try:
        subprocess.run(["kaggle", "datasets", "create", "-p", dataset_dir], check=True)
        print("🎉 Dataset created successfully on Kaggle!")
    except subprocess.CalledProcessError:
        print("⚠️  Dataset already exists or error occurred. Attempting to update...")
        try:
            subprocess.run(["kaggle", "datasets", "version", "-p", dataset_dir, "-m", "Updated realizations"], check=True)
            print("✅ Dataset updated successfully on Kaggle!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to manage Kaggle dataset: {e}")

if __name__ == "__main__":
    # Check if kaggle is configured
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print("❌ Kaggle not configured. Please run setup first.")
    else:
        init_kaggle_dataset()
