# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import argparse
# from azure.ai.ml.entities import Environment
# from utils.azure_client import get_ml_client

# def register_environment(ml_client):
#     env_name = "mle-env"
#     env_path = "config/environment.yaml"

#     custom_env = Environment(
#         name=env_name,
#         description="Environment for MLE project (DKV)",
#         conda_file=env_path,
#         image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
#     )

#     ml_client.environments.create_or_update(custom_env)
#     print(f"✅ Registered environment: {custom_env.name} in workspace: {ml_client.workspace_name}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, default="dev", choices=["dev", "test", "prod"],
#                         help="Environment to target (default: dev)")
#     args = parser.parse_args()

#     ml_client = get_ml_client(args.env)
#     print(f"🔁 Targeting workspace: {ml_client.workspace_name}")  # <== make sure this is present
#     register_environment(ml_client)


import sys
import os
import argparse
from azure.ai.ml.entities import Environment

# Fix import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.azure_client import get_ml_client

def register_environment(ml_client):
    env_name = "mle-env"
    conda_file_path = "config/environment.yaml"

    custom_env = Environment(
        name=env_name,
        description="Environment for MLE project (DKV)",
        conda_file=conda_file_path,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
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
