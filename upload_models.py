from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="backend/models",
    repo_id="Meghla-08/mediscan-models",
    repo_type="model"
)

print("Upload complete")
