# pipeline/run_pipeline.py

from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes

# Authenticate using config.json + interactive browser
credential = InteractiveBrowserCredential(tenant_id="cdddaa56-b77b-4ec9-b63c-83ee2ea3d24a")
ml_client = MLClient.from_config(credential=credential)

print(f"✅ Connected to workspace: {ml_client.workspace_name}")

# Load registered components (latest versions can be dynamically fetched if needed)
preprocess_component = ml_client.components.get(name="preprocess_v2", version="6")
train_component = ml_client.components.get(name="train_model_v1", version="4")
evaluate_component = ml_client.components.get(name="evaluate_model_v1", version="6")

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

# Provide blob path for input data
pipeline_job = credit_default_pipeline(
    input_data=Input(
        type=AssetTypes.URI_FILE,
        path="azureml://subscriptions/4af7b7c7-f338-4dc1-9eed-1752e0d6c8ca/resourcegroups/aniruddh1297-rg/workspaces/credit_defaulter_mle_engineer/datastores/workspaceblobstore/paths/UI/2025-05-27_133844_UTC/default_of_credit_card_clients.xls"
    )
)

pipeline_job.name = "credit-default-pipeline-2"
pipeline_job.display_name = "Credit Card Default ML Pipeline"

# Submit the pipeline job
submitted_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"✅ Pipeline submitted successfully! Job ID: {submitted_job.name}")
