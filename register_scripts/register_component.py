import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
import argparse
from azure.ai.ml import MLClient, load_component
from utils.azure_client import get_ml_client  

def register_components(ml_client):
    component_paths = []
    for root, _, files in os.walk("component_code"):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                component_paths.append(os.path.join(root, file))

    for path in component_paths:
        try:
            print(f"\nRegistering component from: {path}")
            component = load_component(source=path)
            registered_component = ml_client.components.create_or_update(component)
            print(f"Registered: {registered_component.name} (v{registered_component.version})")
        except Exception as e:
            print(f"Failed to register {path}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test", "prod"],
                        help="Target environment for registration (default: dev)")
    args = parser.parse_args()

    ml_client = get_ml_client(args.env)
    print(f"Targeting workspace: {ml_client.workspace_name}")
    register_components(ml_client)


