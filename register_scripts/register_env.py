import argparse
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment

def get_ml_client(env_name="dev"):
    config_path = f".azureml/config.{env_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    credential = DefaultAzureCredential()
    return MLClient.from_config(credential=credential, path=config_path)

def register_environment(ml_client):
    env_name = "mle-env"
    env_path = os.path.join("config", "environment.yaml")

    custom_env = Environment(
        name=env_name,
        description="Environment for MLE project (DKV)",
        conda_file=env_path,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",  # recommended image
    )

    ml_client.environments.create_or_update(custom_env)
    print(f"✅ Registered environment: {custom_env.name} in workspace: {ml_client.workspace_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test", "prod"],
                        help="Environment to target (default: dev)")
    args = parser.parse_args()

    ml_client = get_ml_client(args.env)
    print(f"🔁 Targeting workspace: {ml_client.workspace_name}")
    register_environment(ml_client)
