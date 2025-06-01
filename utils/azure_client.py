import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


if os.getenv("GITHUB_ACTIONS") != "true":
    from dotenv import load_dotenv
    load_dotenv()

def get_ml_client(env_name="dev"):
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "").strip()
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "").strip()
    workspace_key = f"AZURE_WORKSPACE_NAME_{env_name.upper()}"
    workspace_name = os.environ.get(workspace_key, "").strip()

    print(f"::notice::SUBSCRIPTION_ID = {repr(subscription_id)}")
    print(f"::notice::RESOURCE_GROUP  = {repr(resource_group)}")
    print(f"::notice::WORKSPACE_NAME  = {repr(workspace_name)}")

    if not (subscription_id and resource_group and workspace_name):
        raise ValueError("One or more required Azure environment variables are missing or empty.")

    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )


