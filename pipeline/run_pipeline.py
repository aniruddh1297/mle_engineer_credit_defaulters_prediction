from azure.ai.ml import MLClient, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from datetime import datetime
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authenticate using DefaultAzureCredential
try:
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
    logger.info(f"✅ Connected to workspace: {ml_client.workspace_name}")
except Exception as e:
    logger.error("❌ Failed to authenticate with Azure ML Workspace")
    raise e

# Get the latest version of a component
def get_latest_component(name):
    versions = ml_client.components.list(name=name)
    latest = max(versions, key=lambda x: int(x.version))
    logger.info(f"🔄 Using latest version of component '{name}': v{latest.version}")
    return latest

# Get the latest version of an environment
def get_latest_environment(name):
    envs = ml_client.environments.list(name=name)
    latest = max(envs, key=lambda x: int(x.version))
    logger.info(f"🔄 Using latest version of environment '{name}': v{latest.version}")
    return latest

# Get the latest version of a data asset
def get_latest_data_asset(name):
    assets = list(ml_client.data.list(name=name))
    
    if not assets:
        raise ValueError(f"No data assets found with name '{name}'")

    # Sort using natural sort (e.g. v1 < v10 < v100 or date-based v20250529)
    def extract_version_number(asset):
        # Remove any non-digit prefix like 'v'
        match = re.search(r'\d+', asset.version)
        return int(match.group()) if match else 0

    latest = max(assets, key=extract_version_number)
    logger.info(f"🔄 Using latest version of data asset '{name}': {latest.version}")
    return latest

# Load latest components and environments
try:
    preprocess_component = get_latest_component("preprocess_v2")
    train_component = get_latest_component("train_model_v1")
    evaluate_component = get_latest_component("evaluate_model_v1")

    latest_env = get_latest_environment("mle-env")
    preprocess_component.environment = latest_env.id
    train_component.environment = latest_env.id
    evaluate_component.environment = latest_env.id
except Exception as e:
    logger.error("❌ Error loading components or environments")
    raise e

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

# Load latest data asset
try:
    latest_data_asset = get_latest_data_asset("credit_default_data")
    data_input_uri = Input(
        type=AssetTypes.URI_FILE,
        path=f"azureml:{latest_data_asset.name}:{latest_data_asset.version}"
    )
except Exception as e:
    logger.error("❌ Failed to load latest version of data asset")
    raise e

# Submit the pipeline job
try:
    pipeline_job = credit_default_pipeline(input_data=data_input_uri)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    pipeline_job.name = f"credit-default-pipeline-{timestamp}"
    pipeline_job.display_name = f"credit-default-pipeline-{timestamp}"
    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    logger.info(f"✅ Pipeline submitted successfully! Job ID: {submitted_job.name}")
except Exception as e:
    logger.error("❌ Failed to submit pipeline job")
    raise e
