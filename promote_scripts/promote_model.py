# import os
# from azure.ai.ml import MLClient
# from azure.identity import AzureCliCredential
# from azure.ai.ml.entities import Model
# from azure.ai.ml.exceptions import ResourceNotFoundError

# def get_highest_model_version(ml_client, model_name):
#     """Retrieve the highest version of a model registered in the workspace."""
#     models = ml_client.models.list(name=model_name)
#     versions = [int(model.version) for model in models if model.version.isdigit()]
#     if not versions:
#         raise ValueError(f"No registered versions found for model: {model_name}")
#     return str(max(versions))

# def main():
#     credential = AzureCliCredential()

#     # Connect to test and prod workspaces
#     test_ml = MLClient.from_config(credential=credential, path="config.test.json")
#     prod_ml = MLClient.from_config(credential=credential, path="config.prod.json")

#     model_name = "credit-default-model"
#     version = get_highest_model_version(test_ml, model_name)

#     print(f"📦 Fetching model '{model_name}' version '{version}' from test workspace...")

#     # Get and download model from test
#     model = test_ml.models.get(name=model_name, version=version)
#     local_path = model.download()

#     print(f"⬇️  Model downloaded to: {local_path}")
#     print(f"🚀 Promoting model to prod workspace...")

#     # Register model in prod
#     promoted_model = Model(
#         path=local_path,
#         name=model_name,
#         version=version,
#         description="Promoted from test environment",
#         tags={"stage": "production", "source": "test"}
#     )

#     prod_ml.models.create_or_update(promoted_model)

#     print(f"✅ Model '{model_name}' v{version} promoted to prod successfully.")

# if __name__ == "__main__":
#     main()



import os
import sys
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

# ✅ Enable project-level import of utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.azure_client import get_ml_client  # safely loads from .azureml/config.<env>.json


def get_highest_model_version(ml_client, model_name: str) -> str:
    """Returns the highest version number of a registered model."""
    models = ml_client.models.list(name=model_name)
    versions = [int(m.version) for m in models if m.version.isdigit()]
    if not versions:
        raise ValueError(f"❌ No registered versions found for model: {model_name}")
    return str(max(versions))


def main():
    # 🎯 Promote model from TEST ➝ PROD environment
    model_name = "credit-default-model"

    # 🔐 Load clients for both workspaces
    test_ml = get_ml_client("test")
    prod_ml = get_ml_client("prod")

    # 🔍 Get latest version
    version = get_highest_model_version(test_ml, model_name)
    print(f"📦 Fetching model '{model_name}' v{version} from TEST: {test_ml.workspace_name}")

    # ⬇️ Download model
    model = test_ml.models.get(name=model_name, version=version)
    download_dir = os.path.join("promoted_model_downloads", f"{model_name}_v{version}")
    os.makedirs(download_dir, exist_ok=True)

    test_ml.models.download(name=model.name, version=model.version, download_path=download_dir)
    local_model_path = os.path.join(download_dir, model_name)
    print(f"⬇️  Downloaded model to: {local_model_path}")

    # 🚀 Register in PROD
    print(f"🚀 Registering in PROD: {prod_ml.workspace_name}")
    promoted_model = Model(
        path=local_model_path,
        name=model_name,
        version=version,
        description="Promoted from test environment",
        tags={
            "stage": "production",
            "source": "test"
        }
    )
    prod_ml.models.create_or_update(promoted_model)

    print(f"✅ Promotion complete: {model_name} v{version} → PROD")


if __name__ == "__main__":
    main()
