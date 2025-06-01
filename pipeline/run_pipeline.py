import os
import argparse
import re
import logging
from datetime import datetime
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ml_client(env_name="dev"):
    config_path = f".azureml/config.{env_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    credential = DefaultAzureCredential()
    return MLClient.from_config(credential=credential, path=config_path)

def get_latest_component(ml_client, name):
    versions = ml_client.components.list(name=name)
    latest = max(versions, key=lambda x: int(x.version))
    logger.info(f"Using latest version of component '{name}': v{latest.version}")
    return latest

def get_latest_environment(ml_client, name):
    envs = ml_client.environments.list(name=name)
    latest = max(envs, key=lambda x: int(x.version))
    logger.info(f"Using latest version of environment '{name}': v{latest.version}")
    return latest

def get_latest_data_asset(ml_client, name):
    assets = list(ml_client.data.list(name=name))
    if not assets:
        raise ValueError(f"No data assets found with name '{name}'")
    def extract_version_number(asset):
        match = re.search(r'\d+', asset.version)
        return int(match.group()) if match else 0
    latest = max(assets, key=extract_version_number)
    logger.info(f"Using latest version of data asset '{name}': {latest.version}")
    return latest

def define_pipeline(preprocess_component, train_component, evaluate_component):
    @pipeline(default_compute="cpu-cluster")
    def credit_default_pipeline(input_data):
        preprocess_job = preprocess_component(input_data=input_data)
        train_job = train_component(input_data=preprocess_job.outputs.output_path)
        evaluate_job = evaluate_component(
            input_data=preprocess_job.outputs.output_path,
            model_path=train_job.outputs.output_path
        )
        return {"eval_output": evaluate_job.outputs.output_path}
    return credit_default_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test", "prod"],
                        help="Target environment to run the pipeline in (default: dev)")
    args = parser.parse_args()

    ml_client = get_ml_client(args.env)
    logger.info(f"Targeting workspace: {ml_client.workspace_name}")

    try:
        
        preprocess_component = get_latest_component(ml_client, "preprocess_v2")
        train_component = get_latest_component(ml_client, "train_model_v1")
        evaluate_component = get_latest_component(ml_client, "evaluate_model_v1")

        latest_env = get_latest_environment(ml_client, "mle-env")
        for comp in [preprocess_component, train_component, evaluate_component]:
            comp.environment = latest_env.id

    
        latest_data = get_latest_data_asset(ml_client, "credit_default_data")
        data_input_uri = Input(
            type=AssetTypes.URI_FILE,
            path=f"azureml:{latest_data.name}:{latest_data.version}"
        )

        
        credit_pipeline = define_pipeline(preprocess_component, train_component, evaluate_component)
        pipeline_job = credit_pipeline(input_data=data_input_uri)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        pipeline_job.name = f"credit-default-pipeline-{timestamp}"
        pipeline_job.display_name = f"credit-default-pipeline-{timestamp}"
        submitted = ml_client.jobs.create_or_update(pipeline_job)
        logger.info(f"Pipeline submitted successfully! Job ID: {submitted.name}")

    except Exception as e:
        logger.error("Pipeline execution failed")
        raise e
