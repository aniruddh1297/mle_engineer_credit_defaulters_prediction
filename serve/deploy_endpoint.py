# import argparse
# import os
# from azure.identity import DefaultAzureCredential
# from azure.ai.ml import MLClient
# from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration

# def get_ml_client(env_name="dev"):
#     config_path = f".azureml/config.{env_name}.json"
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"❌ Config file not found: {config_path}")
#     credential = DefaultAzureCredential()
#     return MLClient.from_config(credential=credential, path=config_path)

# def deploy_endpoint(ml_client, env_name):
#     endpoint_name = f"credit-default-endpoint-{env_name}"
#     print(f"🚀 Starting deployment to {env_name.upper()} workspace: {ml_client.workspace_name}")

#     # 🧼 Delete existing endpoint if present
#     try:
#         ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
#         print("🗑️ Deleted existing endpoint")
#     except Exception:
#         print("ℹ️ Endpoint does not exist, continuing...")

#     # 🏗️ Create endpoint
#     endpoint = ManagedOnlineEndpoint(
#         name=endpoint_name,
#         description=f"Scoring endpoint for credit default model in {env_name}",
#         auth_mode="key",
#         tags={
#             "purpose": "production" if env_name == "prod" else "staging" if env_name == "test" else "dev",
#             "env": env_name,
#             "owner": "anirudh"
#         }
#     )
#     ml_client.begin_create_or_update(endpoint).result()
#     print("✅ Endpoint created")

#     # 📦 Load latest registered model and environment
#     latest_model = max(ml_client.models.list(name="credit-default-model"), key=lambda m: int(m.version))
#     latest_env = max(ml_client.environments.list(name="mle-env"), key=lambda e: int(e.version))

#     # 🚀 Create deployment
#     deployment = ManagedOnlineDeployment(
#         name="blue",
#         endpoint_name=endpoint_name,
#         model=latest_model,
#         environment=latest_env,
#         code_configuration=CodeConfiguration(
#             code="./serve",
#             scoring_script="score.py"
#         ),
#         instance_type="Standard_E2s_v3",
#         instance_count=2
#     )
#     ml_client.begin_create_or_update(deployment).result()
#     print("🚀 Deployment completed with model v:", latest_model.version)

#     # 🔁 Route traffic
#     endpoint.traffic = {"blue": 100}
#     ml_client.begin_create_or_update(endpoint).result()
#     print("✅ 100% traffic routed to 'blue' deployment")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, choices=["dev", "test", "prod"], default="dev", help="Target environment")
#     args = parser.parse_args()

#     ml_client = get_ml_client(args.env)
#     deploy_endpoint(ml_client, args.env)

# if __name__ == "__main__":
#     main()




# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ✅ add project root to sys.path

# import argparse
# from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration
# from utils.azure_client import get_ml_client  # ✅ centralized client config

# def deploy_endpoint(ml_client, env_name):
#     endpoint_name = f"credit-default-endpoint-{env_name}"
#     print(f"🚀 Starting deployment to {env_name.upper()} workspace: {ml_client.workspace_name}")

#     # 🧼 Delete existing endpoint if present
#     try:
#         ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
#         print("🗑️ Deleted existing endpoint")
#     except Exception:
#         print("ℹ️ Endpoint does not exist, continuing...")

#     # 🏗️ Create endpoint
#     endpoint = ManagedOnlineEndpoint(
#         name=endpoint_name,
#         description=f"Scoring endpoint for credit default model in {env_name}",
#         auth_mode="key",
#         tags={
#             "purpose": "production" if env_name == "prod" else "staging" if env_name == "test" else "dev",
#             "env": env_name,
#             "owner": "anirudh"
#         }
#     )
#     ml_client.begin_create_or_update(endpoint).result()
#     print("✅ Endpoint created")

#     # 📦 Load latest registered model and environment
#     latest_model = max(ml_client.models.list(name="credit-default-model"), key=lambda m: int(m.version))
#     latest_env = max(ml_client.environments.list(name="mle-env"), key=lambda e: int(e.version))

#     # 🚀 Create deployment
#     deployment = ManagedOnlineDeployment(
#         name="blue",
#         endpoint_name=endpoint_name,
#         model=latest_model,
#         environment=latest_env,
#         code_configuration=CodeConfiguration(
#             code="./serve",
#             scoring_script="score.py"
#         ),
#         instance_type="Standard_E2s_v3",
#         instance_count=2
#     )
#     ml_client.begin_create_or_update(deployment).result()
#     print(f"🚀 Deployment completed with model v: {latest_model.version}")

#     # 🔁 Route traffic
#     endpoint.traffic = {"blue": 100}
#     ml_client.begin_create_or_update(endpoint).result()
#     print("✅ 100% traffic routed to 'blue' deployment")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, choices=["dev", "test", "prod"], default="dev",
#                         help="Target environment to deploy to")
#     args = parser.parse_args()

#     ml_client = get_ml_client(args.env)
#     deploy_endpoint(ml_client, args.env)


# import sys
# import os
# import argparse
# from azure.core.exceptions import ResourceNotFoundError
# from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils.azure_client import get_ml_client


# def deploy_endpoint(ml_client, env_name):
#     endpoint_name = f"credit-default-endpoint-{env_name}"
#     print(f"🚀 Starting deployment to {env_name.upper()} workspace: {ml_client.workspace_name}")

#     # 🏗️ Check or create endpoint
#     try:
#         existing_endpoint = ml_client.online_endpoints.get(name=endpoint_name)
#         print(f"ℹ️ Endpoint '{endpoint_name}' already exists. Skipping creation.")
#     except ResourceNotFoundError:
#         print(f"🆕 Endpoint '{endpoint_name}' does not exist. Creating...")
#         endpoint = ManagedOnlineEndpoint(
#             name=endpoint_name,
#             description=f"Scoring endpoint for credit default model in {env_name}",
#             auth_mode="key",
#             tags={
#                 "purpose": "production" if env_name == "prod" else "staging" if env_name == "test" else "dev",
#                 "env": env_name,
#                 "owner": "anirudh"
#             }
#         )
#         ml_client.begin_create_or_update(endpoint).result()
#         print("✅ Endpoint created")

#     # 📦 Get latest model and environment
#     latest_model = max(
#         ml_client.models.list(name="credit-default-model"), key=lambda m: int(m.version)
#     )
#     latest_env = max(
#         ml_client.environments.list(name="mle-env"), key=lambda e: int(e.version)
#     )

#     # 🚀 Create or update deployment
#     deployment = ManagedOnlineDeployment(
#         name="blue",
#         endpoint_name=endpoint_name,
#         model=latest_model,
#         environment=latest_env,
#         code_configuration=CodeConfiguration(
#             code="./serve",
#             scoring_script="score.py"
#         ),
#         instance_type="Standard_E2s_v3",
#         instance_count=2
#     )
#     ml_client.begin_create_or_update(deployment).result()
#     print(f"🚀 Deployment completed with model v: {latest_model.version}")

#     # 🔁 Route 100% traffic to blue
#     ml_client.online_endpoints.begin_update(
#         name=endpoint_name,
#         traffic={"blue": 100}
#     ).result()
#     print("✅ 100% traffic routed to 'blue' deployment")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, choices=["dev", "test", "prod"], default="dev",
#                         help="Target environment to deploy to")
#     args = parser.parse_args()

#     ml_client = get_ml_client(args.env)
#     deploy_endpoint(ml_client, args.env)


import sys
import os
import argparse
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration
)

# 👇 Append project root to path for shared utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.azure_client import get_ml_client


def deploy_endpoint(ml_client, env_name):
    endpoint_name = f"credit-default-endpoint-{env_name}"
    print(f"🚀 Starting deployment to {env_name.upper()} workspace: {ml_client.workspace_name}")

    # 🔍 Check if endpoint exists
    try:
        ml_client.online_endpoints.get(name=endpoint_name)
        print(f"ℹ️ Endpoint '{endpoint_name}' already exists. Skipping creation.")
    except ResourceNotFoundError:
        print(f"🆕 Endpoint '{endpoint_name}' does not exist. Creating...")

        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=f"Scoring endpoint for credit default model in {env_name}",
            auth_mode="key",  # Keep key auth for simplicity
            tags={
                "purpose": "production" if env_name == "prod" else "staging" if env_name == "test" else "dev",
                "env": env_name,
                "owner": "anirudh"
            }
        )
        ml_client.begin_create_or_update(endpoint).result()
        print("✅ Endpoint created")

    # 🧠 Get latest model and environment
    latest_model = max(
        ml_client.models.list(name="credit-default-model"),
        key=lambda m: int(m.version)
    )
    latest_env = max(
        ml_client.environments.list(name="mle-env"),
        key=lambda e: int(e.version)
    )

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
    print(f"🚀 Deployment completed with model v: {latest_model.version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["dev", "test", "prod"], default="dev",
                        help="Target environment to deploy to")
    args = parser.parse_args()

    ml_client = get_ml_client(args.env)
    deploy_endpoint(ml_client, args.env)
