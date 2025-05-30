import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from datetime import datetime

# 🔐 Authenticate
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)
print(f"✅ Connected to workspace: {ml_client.workspace_name}")

# 🗂️ Define your local file
local_path = "data/default_of_credit_card_clients.xls"  
base_name = "credit_default_data"

# 📅 Versioning based on timestamp
timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
version = f"v{timestamp}"

# 📤 Register as versioned Data asset
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
