from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import load_component  # ✅ this is the correct function

# Load workspace config
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Load component from YAML
component = load_component(source="component_code/preprocess/preprocess_component.yml")

# Register it
registered_component = ml_client.components.create_or_update(component)

print(f"✅ Registered: {registered_component.name} (v{registered_component.version})")
