# import os
# from dotenv import load_dotenv
# from azure.identity import DefaultAzureCredential
# from azure.ai.ml import MLClient

# load_dotenv()  # ✅ this loads .env variables into os.environ

# def get_ml_client(env_name="dev"):
#     subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
#     resource_group = os.environ["AZURE_RESOURCE_GROUP"]
#     workspace_key = f"AZURE_WORKSPACE_NAME_{env_name.upper()}"
#     workspace_name = os.environ[workspace_key]

#     credential = DefaultAzureCredential()
#     return MLClient(
#         credential=credential,
#         subscription_id=subscription_id,
#         resource_group_name=resource_group,
#         workspace_name=workspace_name,
#     )


import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

# Only try to load .env locally (optional)
if os.getenv("GITHUB_ACTIONS") != "true":
    from dotenv import load_dotenv
    load_dotenv()

def get_ml_client(env_name="dev"):
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_key = f"AZURE_WORKSPACE_NAME_{env_name.upper()}"
    workspace_name = os.environ.get(workspace_key)

    if not (subscription_id and resource_group and workspace_name):
        raise ValueError(f"❌ Missing Azure environment variables for env: {env_name}")

    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

#test