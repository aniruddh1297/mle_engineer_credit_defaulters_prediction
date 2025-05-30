from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
import os

# Authenticate with DefaultAzureCredential
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

print(f"✅ Connected to workspace: {ml_client.workspace_name}")

# Register custom environment
env_name = "mle-env"
env_path = os.path.join("config", "environment.yaml")

custom_env = Environment(
    name=env_name,
    description="Environment for MLE project (DKV)",
    conda_file=env_path,
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",  # recommended base image
)

ml_client.environments.create_or_update(custom_env)
print(f"✅ Registered environment: {custom_env.name}")
