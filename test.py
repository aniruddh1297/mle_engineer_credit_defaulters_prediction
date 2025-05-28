# from azure.identity import InteractiveBrowserCredential
# from azure.ai.ml import MLClient

# credential = InteractiveBrowserCredential(tenant_id="cdddaa56-b77b-4ec9-b63c-83ee2ea3d24a")

# ml_client = MLClient.from_config(credential=credential)

# print("✅ Connected to:", ml_client.workspace_name)
# print("📦 Listing components...")

# components = ml_client.components.list()
# for c in components:
#     print(f"🔹 {c.name} (v{c.version})")

from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential

# Authenticate using interactive browser login
credential = InteractiveBrowserCredential(tenant_id="cdddaa56-b77b-4ec9-b63c-83ee2ea3d24a")
ml_client = MLClient.from_config(credential=credential)

print(f"✅ Connected to: {ml_client.workspace_name}")
print("📦 Listing components with versions...")

# List all component containers (unique names)
component_containers = ml_client.components.list()

# For each container, list its versions
for container in component_containers:
    versions = ml_client.components.list(name=container.name)
    for v in versions:
        print(f"🔹 {v.name} (v{v.version})")
