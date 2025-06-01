import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import argparse
import urllib.request
import hashlib
from datetime import datetime
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from utils.azure_client import get_ml_client  


def download_google_sheet_as_excel(local_path: str, sheet_id: str):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
        print(f"Downloading spreadsheet as Excel from: {url}")
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded spreadsheet to: {local_path}")
    else:
        print(f"Dataset already exists locally: {local_path}")


def calculate_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def upload_data(ml_client, local_path):
    base_name = "credit_default_data"
    file_hash = calculate_file_hash(local_path)

    print(f"Calculated MD5 hash: {file_hash}")

    existing_assets = list(ml_client.data.list(name=base_name))
    for asset in existing_assets:
        if asset.tags and asset.tags.get("hash") == file_hash:
            print(f"Identical dataset already registered: {asset.name}:{asset.version} — skipping upload.")
            return

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    version = f"v{timestamp}"

    data_asset = Data(
        name=base_name,
        version=version,
        description="Credit card default dataset uploaded for pipeline input",
        path=local_path,
        type=AssetTypes.URI_FILE,
        tags={"hash": file_hash},
    )

    ml_client.data.create_or_update(data_asset)
    print(f"Registered new data asset: {base_name}:{version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test", "prod"],
                        help="Target environment to upload data to (default: dev)")
    parser.add_argument("--use_drive", action="store_true",
                        help="Download from Google Sheets if the file is missing")
    args = parser.parse_args()

    local_path = "data/default_of_credit_card_clients.xls"
    sheet_id = "1TW8rGUH07tUJMhsjlCwVKLUh1zOrTX3h"

    if args.use_drive:
        download_google_sheet_as_excel(local_path, sheet_id)

    ml_client = get_ml_client(args.env)
    print(f"Connected to workspace: {ml_client.workspace_name}")
    upload_data(ml_client, local_path)

