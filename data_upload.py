import os
import argparse
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

def upload_data(ml_client):
    # 🗂️ Define your local file and name
    local_path = "data/default_of_credit_card_clients.xls"  
    base_name = "credit_default_data"

    # 📅 Timestamp-based versioning
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    version = f"v{timestamp}"

    data_asset = Data(
        name=base_name,
        version=version,
        description="Credit card default dataset uploaded for pipeline input",
        path=local_path,
        type=AssetTypes.URI_FILE,
    )

    # ✅ Upload and register
    ml_client.data.create_or_update(data_asset)
    print(f"📦 Registered data asset: {base_name}:{version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test", "prod"],
                        help="Target environment to upload data to (default: dev)")
    args = parser.parse_args()

    client = get_ml_client(args.env)
    print(f"✅ Connected to workspace: {client.workspace_name}")
    upload_data(client)
