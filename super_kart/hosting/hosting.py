from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="super_kart/deployment",
    repo_id="nirmalhugface/super_kart_sales_forecasting",
    repo_type="space",
    path_in_repo="",
)
