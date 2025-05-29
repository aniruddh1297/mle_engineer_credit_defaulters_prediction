# from azure.identity import AzureCliCredential
# from azure.ai.ml import MLClient, Input
# from azure.ai.ml.dsl import pipeline
# from azure.ai.ml.constants import AssetTypes

# # Authenticate using Azure CLI (non-interactive)
# credential = AzureCliCredential()
# ml_client = MLClient.from_config(credential=credential)

# print(f"✅ Connected to workspace: {ml_client.workspace_name}")

# # Helper function to get the latest version of a component
# def get_latest_component(name):
#     versions = ml_client.components.list(name=name)
#     latest = max(versions, key=lambda x: int(x.version))  # assumes version is integer
#     print(f"🔄 Using latest version of {name}: v{latest.version}")
#     return latest

# # Load components dynamically
# preprocess_component = get_latest_component("preprocess_v2")
# train_component = get_latest_component("train_model_v1")
# evaluate_component = get_latest_component("evaluate_model_v1")

# # Define pipeline
# @pipeline(default_compute="cpu-cluster")
# def credit_default_pipeline(input_data):
#     preprocess_job = preprocess_component(input_data=input_data)
#     train_job = train_component(input_data=preprocess_job.outputs.output_path)
#     evaluate_job = evaluate_component(
#         input_data=preprocess_job.outputs.output_path,
#         model_path=train_job.outputs.output_path
#     )
#     return {"eval_output": evaluate_job.outputs.output_path}

# # Dynamically link to the uploaded dataset
# pipeline_job = credit_default_pipeline(
#     input_data=Input(
#         type=AssetTypes.URI_FILE,
#         path="azureml://subscriptions/4af7b7c7-f338-4dc1-9eed-1752e0d6c8ca/resourcegroups/aniruddh1297-rg/workspaces/credit_defaulter_mle_engineer/datastores/workspaceblobstore/paths/UI/2025-05-27_133844_UTC/default_of_credit_card_clients.xls"
#     )
# )

# pipeline_job.name = "credit-default-pipeline"
# pipeline_job.display_name = "Credit Card Default ML Pipeline (Auto Component Version)"

# # Submit pipeline
# submitted_job = ml_client.jobs.create_or_update(pipeline_job)
# print(f"✅ Pipeline submitted successfully! Job ID: {submitted_job.name}")


from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
from datetime import datetime

# Authenticate non-interactively via Azure CLI
credential = AzureCliCredential()
ml_client = MLClient.from_config(credential=credential)

print(f"✅ Connected to workspace: {ml_client.workspace_name}")

# Helper to get latest version of a component
def get_latest_component(name):
    versions = ml_client.components.list(name=name)
    latest = max(versions, key=lambda x: int(x.version))  # assumes version is an integer
    print(f"🔄 Using latest version of {name}: v{latest.version}")
    return latest

# Load latest registered components
preprocess_component = get_latest_component("preprocess_v2")
train_component = get_latest_component("train_model_v1")
evaluate_component = get_latest_component("evaluate_model_v1")

def get_latest_environment(name):
    envs = ml_client.environments.list(name=name)
    latest = max(envs, key=lambda x: int(x.version))
    print(f"🔄 Using latest version of environment '{name}': v{latest.version}")
    return latest

# Step 2: After loading components
latest_env = get_latest_environment("mle-env")
preprocess_component.environment = latest_env.id
train_component.environment = latest_env.id
evaluate_component.environment = latest_env.id

# Define pipeline
@pipeline(default_compute="cpu-cluster")
def credit_default_pipeline(input_data):
    preprocess_job = preprocess_component(input_data=input_data)
    train_job = train_component(input_data=preprocess_job.outputs.output_path)
    evaluate_job = evaluate_component(
        input_data=preprocess_job.outputs.output_path,
        model_path=train_job.outputs.output_path
    )
    return {"eval_output": evaluate_job.outputs.output_path}

# Use blob path for input data
pipeline_job = credit_default_pipeline(
    input_data=Input(
        type=AssetTypes.URI_FILE,
        path="azureml://subscriptions/4af7b7c7-f338-4dc1-9eed-1752e0d6c8ca/resourcegroups/aniruddh1297-rg/workspaces/credit_defaulter_mle_engineer/datastores/workspaceblobstore/paths/UI/2025-05-27_133844_UTC/default_of_credit_card_clients.xls"
    )
)

# Generate unique pipeline job name with timestamp
timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
pipeline_job.name = f"credit-default-pipeline-{timestamp}"
pipeline_job.display_name = f"credit-default-pipeline-{timestamp}"

# Submit the pipeline
submitted_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"✅ Pipeline submitted successfully! Job ID: {submitted_job.name}")
