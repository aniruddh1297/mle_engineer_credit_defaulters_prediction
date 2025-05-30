# # serve/deploy_endpoint.py

# import json
# import logging
# from azure.ai.ml import MLClient
# from azure.identity import DefaultAzureCredential
# from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Auth & client setup
# credential = DefaultAzureCredential()
# ml_client = MLClient.from_config(credential=credential)
# logger.info(f"✅ Connected to workspace: {ml_client.workspace_name}")

# # Find the latest model version
# def get_latest_model(name):
#     models = ml_client.models.list(name=name)
#     latest = max(models, key=lambda m: m.version)
#     logger.info(f"📦 Using latest model version: {latest.version}")
#     return latest

# model = get_latest_model("credit-default-model")  # Adjust if your model has a different name

# # Define endpoint name
# endpoint_name = "credit-default-endpoint"

# # Create endpoint (idempotent)
# endpoint = ManagedOnlineEndpoint(
#     name=endpoint_name,
#     description="Real-time scoring endpoint for credit default prediction",
#     auth_mode="key"
# )

# try:
#     ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
#     logger.info(f"🚀 Endpoint created/updated: {endpoint_name}")
# except Exception as e:
#     logger.error("❌ Failed to create endpoint")
#     raise e

# # Define deployment
# deployment = ManagedOnlineDeployment(
#     name="blue",
#     endpoint_name=endpoint_name,
#     model=model.id,
#     environment="azureml:mle-env@latest",
#     code_path="serve",
#     scoring_script="score.py",
#     instance_type="Standard_E2s_v3",
#     instance_count=1
# )

# # Deploy model
# try:
#     ml_client.online_deployments.begin_create_or_update(deployment).wait()
#     logger.info("✅ Deployment successful")

#     # Set traffic to blue
#     ml_client.online_endpoints.begin_update(
#         endpoint_name=endpoint_name,
#         traffic={"blue": 100}
#     ).wait()
#     logger.info("🌐 Traffic routed to blue deployment")
# except Exception as e:
#     logger.error("❌ Deployment failed")
#     raise e


# serve/deploy_endpoint.py

import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authenticate and create ML client
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)
logger.info(f"✅ Connected to workspace: {ml_client.workspace_name}")

# Fetch the latest registered model
def get_latest_model(name):
    models = ml_client.models.list(name=name)
    latest = max(models, key=lambda m: int(m.version))
    logger.info(f"📦 Using latest model version: {latest.version}")
    return latest

# Fetch the latest registered environment
def get_latest_environment(name):
    envs = ml_client.environments.list(name=name)
    latest = max(envs, key=lambda e: int(e.version))
    logger.info(f"🔄 Using latest environment version: {latest.version}")
    return latest

# Load latest model and environment
model = get_latest_model("credit-default-model")
environment = get_latest_environment("mle-env")

# Define the endpoint name
endpoint_name = "credit-default-endpoint-v2"

# Define endpoint (idempotent creation)
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Real-time scoring endpoint for credit default prediction",
    auth_mode="key"
)

try:
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    logger.info(f"🚀 Endpoint created/updated: {endpoint_name}")
except Exception as e:
    logger.error("❌ Failed to create endpoint")
    raise e

# Define deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model.id,
    environment=environment.id,  # Use the full resource ID
    code_path="serve",
    scoring_script="score.py",
    instance_type="Standard_E2s_v3",
    instance_count=1
)

# Deploy the model to the endpoint
try:
    ml_client.online_deployments.begin_create_or_update(deployment).wait()
    logger.info("✅ Deployment successful")

    # Route 100% traffic to this deployment
    ml_client.online_endpoints.begin_update(
        endpoint_name=endpoint_name,
        traffic={"blue": 100}
    ).wait()
    logger.info("🌐 Traffic routed to blue deployment")
except Exception as e:
    logger.error("❌ Deployment failed")
    raise e
