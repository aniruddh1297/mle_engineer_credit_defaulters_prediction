# # ✅ serve/deploy_endpoint.py
# import uuid
# from azure.identity import DefaultAzureCredential
# from azure.ai.ml import MLClient
# from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
# from azure.ai.ml.entities import CodeConfiguration
# import os

# # 🔐 Authenticate using config from parent folder
# credential = DefaultAzureCredential()
# ml_client = MLClient.from_config(credential=credential, path="config/config.dev.json")

# # 🎯 Define endpoint name
# endpoint_name = "credit-default-endpoint"

# # 🧼 Delete if exists
# try:
#     ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
#     print("🗑️ Deleted existing endpoint")
# except Exception:
#     print("ℹ️ Endpoint does not exist, continuing...")

# # 🚀 Create endpoint
# endpoint = ManagedOnlineEndpoint(
#     name=endpoint_name,
#     description="Online scoring endpoint for credit default model",
#     auth_mode="key",
#     tags={
#         "purpose": "production",
#         "owner": "anirudh",
#         "env": "prod"
#     }
# )
# ml_client.begin_create_or_update(endpoint).result()
# print("✅ Endpoint created")

# # Get latest model
# model_versions = list(ml_client.models.list(name="credit-default-model"))
# latest_model = max(model_versions, key=lambda m: int(m.version))

# # Get latest environment
# env_versions = list(ml_client.environments.list(name="mle-env"))
# latest_env = max(env_versions, key=lambda e: int(e.version))

# # Create deployment
# deployment = ManagedOnlineDeployment(
#     name="blue",
#     endpoint_name=endpoint_name,
#     model=latest_model,
#     environment=latest_env,
#     code_configuration=CodeConfiguration(
#         code="./serve", 
#         scoring_script="score.py"
#     ),
#     instance_type="Standard_E2s_v3",
#     instance_count=2
# )
# ml_client.begin_create_or_update(deployment).result()
# print("🚀 Deployment completed")

# # 🔁 Route traffic to blue deployment
# endpoint.traffic = {"blue": 100}
# ml_client.begin_create_or_update(endpoint).result()
# print("✅ Traffic routed to deployment 'blue'")

import argparse
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration

def get_ml_client(env_name="dev"):
    config_path = f".azureml/config.{env_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    credential = DefaultAzureCredential()
    return MLClient.from_config(credential=credential, path=config_path)

def deploy_endpoint(ml_client, env_name):
    endpoint_name = f"credit-default-endpoint-{env_name}"
    print(f"🚀 Starting deployment to {env_name.upper()} workspace: {ml_client.workspace_name}")

    # 🧼 Delete existing endpoint if present
    try:
        ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
        print("🗑️ Deleted existing endpoint")
    except Exception:
        print("ℹ️ Endpoint does not exist, continuing...")

    # 🏗️ Create endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description=f"Scoring endpoint for credit default model in {env_name}",
        auth_mode="key",
        tags={
            "purpose": "production" if env_name == "prod" else "staging",
            "env": env_name,
            "owner": "anirudh"
        }
    )
    ml_client.begin_create_or_update(endpoint).result()
    print("✅ Endpoint created")

    # 📦 Load latest registered model and environment
    latest_model = max(ml_client.models.list(name="credit-default-model"), key=lambda m: int(m.version))
    latest_env = max(ml_client.environments.list(name="mle-env"), key=lambda e: int(e.version))

    # 🚀 Create deployment
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=latest_model,
        environment=latest_env,
        code_configuration=CodeConfiguration(
            code="./serve",
            scoring_script="score.py"
        ),
        instance_type="Standard_E2s_v3",
        instance_count=2
    )
    ml_client.begin_create_or_update(deployment).result()
    print("🚀 Deployment completed with model v:", latest_model.version)

    # 🔁 Route traffic
    endpoint.traffic = {"blue": 100}
    ml_client.begin_create_or_update(endpoint).result()
    print("✅ 100% traffic routed to 'blue' deployment")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["dev", "test", "prod"], default="dev", help="Target environment")
    args = parser.parse_args()

    ml_client = get_ml_client(args.env)
    deploy_endpoint(ml_client, args.env)

if __name__ == "__main__":
    main()
