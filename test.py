from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
assets = ml_client.data.list(name="credit_default_data")

for asset in assets:
    print(asset.name, asset.version, asset.tags)