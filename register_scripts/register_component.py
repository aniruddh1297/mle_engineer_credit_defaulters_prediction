import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_component

# Authenticate using DefaultAzureCredential (supports CLI, VS Code, managed identity, etc.)
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)
print("✅ Connected to Azure ML Workspace")

# Collect all component YAML files from the component_code folder
component_paths = []
for root, _, files in os.walk("component_code"):
    for file in files:
        if file.endswith(".yaml") or file.endswith(".yml"):
            component_paths.append(os.path.join(root, file))

# Register each component
for path in component_paths:
    try:
        print(f"\n📦 Registering component from: {path}")
        component = load_component(source=path)
        registered_component = ml_client.components.create_or_update(component)
        print(f"✅ Registered: {registered_component.name} (v{registered_component.version})")
    except Exception as e:
        print(f"❌ Failed to register {path}: {str(e)}")
