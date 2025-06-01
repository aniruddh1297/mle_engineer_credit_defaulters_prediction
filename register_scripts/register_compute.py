import sys
import os
import argparse
import yaml
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceExistsError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.azure_client import get_ml_client


def register_compute(ml_client, config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Compute config not found at: {config_path}")

    with open(config_path, "r") as f:
        compute_config = yaml.safe_load(f)

    compute_name = compute_config.get("name", "cpu-cluster")
    vm_size = compute_config.get("vm_size", "Standard_E2s_v3")
    min_instances = compute_config.get("min_instances", 0)
    max_instances = compute_config.get("max_instances", 2)
    idle_seconds = compute_config.get("idle_time_before_scale_down", 120)

    try:
        ml_client.compute.get(compute_name)
        print(f"Compute cluster '{compute_name}' already exists.")
    except Exception:
        cpu_cluster = AmlCompute(
            name=compute_name,
            size=vm_size,
            min_instances=min_instances,
            max_instances=max_instances,
            idle_time_before_scale_down=idle_seconds,
            tier="Standard"
        )
        ml_client.compute.begin_create_or_update(cpu_cluster).result()
        print(f"Registered compute cluster: {compute_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test", "prod"],
                        help="Environment to target (default: dev)")
    parser.add_argument("--config", type=str, default="config/compute.yaml",
                        help="Path to compute configuration YAML file")
    args = parser.parse_args()

    ml_client = get_ml_client(args.env)
    print(f"Targeting workspace: {ml_client.workspace_name}")
    register_compute(ml_client, args.config)
