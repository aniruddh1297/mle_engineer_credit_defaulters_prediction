import os
import argparse
import urllib.request
from datetime import datetime
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

def get_ml_client(env_name="dev"):
    config_path = f".azureml/config.{env_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    credential = DefaultAzureCredential()
    return MLClient.from_config(credential=credential, path=config_path)

def download_google_sheet_as_excel(local_path: str, sheet_id: str):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
        print(f"⬇️ Downloading spreadsheet as Excel from: {url}")
        urllib.request.urlretrieve(url, local_path)
        print(f"✅ Downloaded spreadsheet to: {local_path}")
    else:
        print(f"✅ Dataset already exists locally: {local_path}")

def upload_data(ml_client, local_path):
    base_name = "credit_default_data"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    version = f"v{timestamp}"

    data_asset = Data(
        name=base_name,
        version=version,
        description="Credit card default dataset uploaded for pipeline input",
        path=local_path,
        type=AssetTypes.URI_FILE,
    )

    ml_client.data.create_or_update(data_asset)
    print(f"📦 Registered data asset: {base_name}:{version}")

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

    client = get_ml_client(args.env)
    print(f"✅ Connected to workspace: {client.workspace_name}")
    upload_data(client, local_path)
