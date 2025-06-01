import sys
import os
import argparse
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.azure_client import get_ml_client

def deploy_endpoint(ml_client, env_name):
    endpoint_name = f"credit-default-endpoint-{env_name}"
    print(f"🚀 Starting deployment to {env_name.upper()} workspace: {ml_client.workspace_name}")

    try:
        ml_client.online_endpoints.get(name=endpoint_name)
        print(f"Endpoint '{endpoint_name}' already exists. Skipping creation.")
    except ResourceNotFoundError:
        print(f"Endpoint '{endpoint_name}' does not exist. Creating...")

        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=f"Scoring endpoint for credit default model in {env_name}",
            auth_mode="key",  
            tags={
                "purpose": "production" if env_name == "prod" else "staging" if env_name == "test" else "dev",
                "env": env_name,
                "owner": "anirudh"
            }
        )
        ml_client.begin_create_or_update(endpoint).result()
        print("Endpoint created")

    latest_model = max(
        ml_client.models.list(name="credit-default-model"),
        key=lambda m: int(m.version)
    )
    latest_env = max(
        ml_client.environments.list(name="mle-env"),
        key=lambda e: int(e.version)
    )

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
    print(f"Deployment completed with model v: {latest_model.version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["dev", "test", "prod"], default="dev",
                        help="Target environment to deploy to")
    args = parser.parse_args()

    ml_client = get_ml_client(args.env)
    deploy_endpoint(ml_client, args.env)
